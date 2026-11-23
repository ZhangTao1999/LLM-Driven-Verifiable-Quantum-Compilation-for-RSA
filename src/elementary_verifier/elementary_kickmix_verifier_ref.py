"""Zero-trust elementary-gate kickmix reference verifier (pure Python).

Implements the exact EGKMV1 binary stream protocol with ZERO knowledge of RNS,
primes, QROMs, adders, comparators, or phase tables. Accepts ONLY 9 flat 1-bit
wire opcodes:
  0: OP_EOF
  1: OP_ALLOC(w)
  2: OP_FREE_ZERO(w)
  3: OP_X(w)
  4: OP_CX(c, t)
  5: OP_Z(w)
  6: OP_CZ(a, b)
  7: OP_GLOBAL_Z()
  8: OP_AND(c1, c2, t)   [2 uint32 words; requires live clean |0> target]
  9: OP_MX_FREE(w, m)    [samples m from verifier SplitMix64 PRNG, kicks Z if m=1]
"""
from __future__ import annotations

import json
import struct
import sys
from typing import BinaryIO

MAGIC = b"EGKMV1\x00\x00"
MASK64 = 0xFFFFFFFFFFFFFFFF


class GateViolationError(RuntimeError):
    """Raised immediately when any elementary-gate or cleanup invariant fails."""


class SplitMix64:
    """Deterministic 64-bit PRNG identical across C++, C#, and Python verifiers."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def next_bit(self) -> int:
        return self.next_u64() & 1


def _read_exact(stream: BinaryIO, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise GateViolationError("TRUNCATED_STREAM_WITHOUT_EOF")
        buf.extend(chunk)
    return bytes(buf)


def verify_egkm_stream(stream: BinaryIO) -> dict:
    hdr = _read_exact(stream, 28)
    stream_bytes = 28
    magic, mx_seed, num_branches, max_wires, num_init_wires = struct.unpack("<8sQIII", hdr)
    if magic != MAGIC:
        raise GateViolationError(f"Invalid magic header: {magic!r}")
    if not (1 <= num_branches <= 64):
        raise GateViolationError(f"Invalid num_branches={num_branches}")
    if not (1 <= max_wires <= 16384):
        raise GateViolationError(f"Invalid max_wires={max_wires}")
    if num_init_wires > max_wires:
        raise GateViolationError(f"Invalid num_init_wires={num_init_wires} > max_wires={max_wires}")

    branch_mask = MASK64 if num_branches == 64 else ((1 << num_branches) - 1)
    rng = SplitMix64(mx_seed)

    live = [False] * max_wires
    val = [0] * max_wires
    is_init = [False] * max_wires
    readonly_init: list[tuple[int, int]] = []

    live_wires = 0
    peak_live_wires = 0

    for _ in range(num_init_wires):
        w, flags, raw_bp = struct.unpack("<HHQ", _read_exact(stream, 12))
        stream_bytes += 12
        if (flags & ~1) != 0:
            raise GateViolationError(f"NONZERO_HEADER_FLAGS flags={flags}")
        if (raw_bp & ~branch_mask) != 0:
            raise GateViolationError(f"NONZERO_UNUSED_BRANCH_BITS w={w}")
        if w >= max_wires or live[w]:
            raise GateViolationError(f"Invalid or duplicate init wire {w}")
        masked_bp = raw_bp & branch_mask
        live[w] = True
        is_init[w] = True
        val[w] = masked_bp
        live_wires += 1
        if live_wires > peak_live_wires:
            peak_live_wires = live_wires
        if flags & 1:
            readonly_init.append((w, masked_bp))

    (num_output_groups,) = struct.unpack("<I", _read_exact(stream, 4))
    stream_bytes += 4
    if num_output_groups > max_wires:
        raise GateViolationError(f"Invalid num_output_groups={num_output_groups} > max_wires={max_wires}")
    output_groups: list[list[int]] = []
    for _ in range(num_output_groups):
        (glen,) = struct.unpack("<I", _read_exact(stream, 4))
        stream_bytes += 4
        if not (1 <= glen <= 64):
            raise GateViolationError(f"Invalid output group length {glen}")
        gwires = list(struct.unpack(f"<{glen}H", _read_exact(stream, 2 * glen)))
        stream_bytes += 2 * glen
        for gw in gwires:
            if gw >= max_wires or not is_init[gw]:
                raise GateViolationError(f"Output group wire {gw} not in init wires")
        output_groups.append(gwires)

    phase_sign = 0
    n_alloc = 0
    n_free_zero = 0
    n_x = 0
    n_cx = 0
    n_z = 0
    n_cz = 0
    n_global_z = 0
    n_and = 0
    n_mx = 0

    # Read instructions in buffered chunks
    pending = b""
    eof_seen = False
    while not eof_seen:
        chunk = stream.read(65536)
        if not chunk:
            raise GateViolationError("TRUNCATED_STREAM_WITHOUT_EOF")
        stream_bytes += len(chunk)
        data = pending + chunk
        n_words = len(data) // 4
        pending = data[n_words * 4 :]
        words = struct.unpack(f"<{n_words}I", data[: n_words * 4])
        idx = 0
        while idx < n_words:
            word = words[idx]
            idx += 1
            op = word & 0xF
            if op == 4:  # OP_CX (most frequent)
                c = (word >> 4) & 0x3FFF
                t = (word >> 18) & 0x3FFF
                if c >= max_wires or t >= max_wires or not live[c] or not live[t] or c == t:
                    raise GateViolationError(f"INVALID_CX c={c} t={t}")
                val[t] ^= val[c]
                n_cx += 1
            elif op == 6:  # OP_CZ
                a = (word >> 4) & 0x3FFF
                b = (word >> 18) & 0x3FFF
                if a >= max_wires or b >= max_wires or not live[a] or not live[b] or a == b:
                    raise GateViolationError(f"INVALID_CZ a={a} b={b}")
                phase_sign ^= val[a] & val[b]
                n_cz += 1
            elif op == 3:  # OP_X
                if (word >> 18) != 0:
                    raise GateViolationError("NONZERO_RESERVED_BITS op=X")
                w = (word >> 4) & 0x3FFF
                if w >= max_wires or not live[w]:
                    raise GateViolationError(f"INVALID_X w={w}")
                val[w] ^= branch_mask
                n_x += 1
            elif op == 5:  # OP_Z
                if (word >> 18) != 0:
                    raise GateViolationError("NONZERO_RESERVED_BITS op=Z")
                w = (word >> 4) & 0x3FFF
                if w >= max_wires or not live[w]:
                    raise GateViolationError(f"INVALID_Z w={w}")
                phase_sign ^= val[w]
                n_z += 1
            elif op == 8:  # OP_AND (2 words)
                if idx >= n_words:
                    extra = _read_exact(stream, 4 - len(pending))
                    stream_bytes += len(extra)
                    w2_bytes = pending + extra
                    pending = b""
                    (word2,) = struct.unpack("<I", w2_bytes)
                else:
                    word2 = words[idx]
                    idx += 1
                if (word2 >> 14) != 0:
                    raise GateViolationError("NONZERO_RESERVED_BITS op=AND")
                c1 = (word >> 4) & 0x3FFF
                c2 = (word >> 18) & 0x3FFF
                t = word2 & 0x3FFF
                if c1 >= max_wires or c2 >= max_wires or t >= max_wires:
                    raise GateViolationError(f"INVALID_AND_RANGE c1={c1} c2={c2} t={t}")
                if not live[c1] or not live[c2] or not live[t]:
                    raise GateViolationError(f"DEAD_WIRE_IN_AND c1={c1} c2={c2} t={t}")
                if c1 == c2 or t == c1 or t == c2:
                    raise GateViolationError(f"WIRE_ALIAS_IN_AND c1={c1} c2={c2} t={t}")
                if (val[t] & branch_mask) != 0:
                    raise GateViolationError(f"DIRTY_AND_TARGET t={t} val=0x{val[t] & branch_mask:016x}")
                val[t] = val[c1] & val[c2]
                n_and += 1
            elif op == 9:  # OP_MX_FREE
                if (word >> 19) != 0:
                    raise GateViolationError("NONZERO_RESERVED_BITS op=MX_FREE")
                w = (word >> 4) & 0x3FFF
                expected_m = (word >> 18) & 1
                if w >= max_wires or not live[w]:
                    raise GateViolationError(f"INVALID_MX w={w}")
                m = rng.next_bit()
                if m != expected_m:
                    raise GateViolationError(f"MX_SYNC_MISMATCH w={w} expected={expected_m} actual={m}")
                if m:
                    phase_sign ^= val[w]
                val[w] = 0
                live[w] = False
                live_wires -= 1
                n_mx += 1
            elif op == 1:  # OP_ALLOC
                if (word >> 18) != 0:
                    raise GateViolationError("NONZERO_RESERVED_BITS op=ALLOC")
                w = (word >> 4) & 0x3FFF
                if w >= max_wires or live[w]:
                    raise GateViolationError(f"INVALID_ALLOC w={w}")
                live[w] = True
                val[w] = 0
                live_wires += 1
                if live_wires > peak_live_wires:
                    peak_live_wires = live_wires
                n_alloc += 1
            elif op == 2:  # OP_FREE_ZERO
                if (word >> 18) != 0:
                    raise GateViolationError("NONZERO_RESERVED_BITS op=FREE_ZERO")
                w = (word >> 4) & 0x3FFF
                if w >= max_wires or not live[w]:
                    raise GateViolationError(f"INVALID_FREE w={w}")
                if (val[w] & branch_mask) != 0:
                    raise GateViolationError(f"NONZERO_FREE w={w} val=0x{val[w] & branch_mask:016x}")
                live[w] = False
                val[w] = 0
                live_wires -= 1
                n_free_zero += 1
            elif op == 7:  # OP_GLOBAL_Z
                if (word >> 4) != 0:
                    raise GateViolationError("NONZERO_RESERVED_BITS op=GLOBAL_Z")
                phase_sign ^= branch_mask
                n_global_z += 1
            elif op == 0:  # OP_EOF
                if word != 0:
                    raise GateViolationError(f"NONZERO_EOF_WORD word=0x{word:08x}")
                if idx != n_words or len(pending) != 0:
                    raise GateViolationError("TRAILING_BYTES_AFTER_EOF")
                if stream.read(1):
                    raise GateViolationError("TRAILING_BYTES_AFTER_EOF")
                eof_seen = True
                break
            else:
                raise GateViolationError(f"UNKNOWN_OPCODE op={op}")

    # End-of-stream zero-trust checks
    if live_wires != num_init_wires:
        raise GateViolationError(f"LEAKED_OR_MISSING_WIRES live={live_wires} expected={num_init_wires}")
    for w in range(max_wires):
        if live[w] != is_init[w]:
            raise GateViolationError(f"FINAL_LIVENESS_MISMATCH w={w} live={live[w]} expected={is_init[w]}")
    for w, init_bp in readonly_init:
        if (val[w] & branch_mask) != init_bp:
            raise GateViolationError(f"READONLY_INPUT_MUTATED w={w}")
    if (phase_sign & branch_mask) != 0:
        raise GateViolationError(f"NONZERO_RESIDUAL_PHASE phase_mask=0x{phase_sign & branch_mask:016x}")

    outputs_by_group: list[list[int]] = []
    for gwires in output_groups:
        branch_ints = [0] * num_branches
        for bit_idx, gw in enumerate(gwires):
            bp = val[gw] & branch_mask
            if bp:
                for b in range(num_branches):
                    if (bp >> b) & 1:
                        branch_ints[b] |= 1 << bit_idx
        outputs_by_group.append(branch_ints)

    total_ops = n_alloc + n_free_zero + n_x + n_cx + n_z + n_cz + n_global_z + n_and + n_mx
    return {
        "status": "PASS",
        "verifier": "python_reference_egkmv1",
        "num_branches": num_branches,
        "peak_live_wires": peak_live_wires,
        "final_live_wires": live_wires,
        "stream_bytes": stream_bytes,
        "residual_phase_mask_u64": f"0x{phase_sign & branch_mask:016x}",
        "final_rng_state_u64": f"0x{rng.state:016x}",
        "gate_counts": {
            "alloc": n_alloc,
            "free_zero": n_free_zero,
            "X": n_x,
            "CX": n_cx,
            "Z": n_z,
            "CZ": n_cz,
            "global_Z": n_global_z,
            "AND": n_and,
            "MX": n_mx,
            "total_stream_ops": total_ops,
        },
        "output_groups": outputs_by_group,
    }


def main() -> int:
    try:
        res = verify_egkm_stream(sys.stdin.buffer)
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()
        return 0
    except GateViolationError as exc:
        err = {"status": "FAIL", "error": str(exc)}
        sys.stdout.write(json.dumps(err) + "\n")
        sys.stdout.flush()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
