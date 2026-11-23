"""Independent OPSTRM01 binary stream reader and randomized SHAKE-256 kickmix engine (O(1) RAM).

Zero dependency on ``flat_stream.Simulator``: decodes the OPSTRM01 wire format and executes
basis permutations, clean-ancilla lifetime tracking, and randomized X-measurement phase
kickback through a completely separate execution engine for local dual-backend verification.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
import struct
from typing import Protocol, Sequence

# Self-contained OPSTRM01 opcode table (zero import from flat_stream.Simulator)
_OP_X = 0
_OP_Z = 1
_OP_NEG = 2
_OP_CX = 3
_OP_CZ = 4
_OP_SWAP = 5
_OP_CCX = 6
_OP_HMR = 9
_OP_OXOR = 10
_OP_AND_COMP = 20
_OP_AND_REL = 21
_OP_QALLOC = 22
_OP_QFREE = 23
_OP_OCLEAR = 24
_OP_CSWAP = 25

EXPECTED_ARITY: dict[int, int] = {
    _OP_QALLOC: 1,
    _OP_QFREE: 1,
    _OP_X: 1,
    _OP_CX: 2,
    _OP_SWAP: 2,
    _OP_CCX: 3,
    _OP_CSWAP: 3,
    _OP_Z: 2,
    _OP_CZ: 3,
    _OP_NEG: 1,
    _OP_HMR: 2,
    _OP_AND_COMP: 5,
    _OP_AND_REL: 5,
    _OP_OCLEAR: 1,
    _OP_OXOR: 2,
}


@dataclass
class TestCase:
    inputs: list[int]
    expected: list[int]
    expect_fail: bool = False


class TaskProtocol(Protocol):
    """Interface required by ``run_task`` when driving streaming kickmix evaluations."""

    num_tests: int
    seed_domain: bytes

    def gen_test_indexed(self, index: int, rng) -> TestCase: ...


def _read_exact(fh, n: int) -> bytes:
    buf = fh.read(n)
    if len(buf) != n:
        raise ValueError(f"Truncated OPSTRM01 stream: expected {n} bytes, got {len(buf)}")
    return buf


def _read_header(fh):
    n_regs, n_ops = struct.unpack("<II", _read_exact(fh, 8))
    regs = []
    for _ in range(n_regs):
        (w,) = struct.unpack("<I", _read_exact(fh, 4))
        wires = list(struct.unpack(f"<{w}I", _read_exact(fh, 4 * w))) if w else []
        regs.append((w, wires))
    return regs, n_ops


def _extract_expected_widths(task: TaskProtocol) -> list[int] | None:
    rw = getattr(task, "register_widths", None)
    if rw is not None:
        return [int(w) for w in rw]
    schema_fn = getattr(task, "schema", None)
    if callable(schema_fn):
        specs = schema_fn()
        if specs is not None:
            return [int(getattr(s, "width", s)) for s in specs]
    return None


def _validate_register_schema(
    regs: Sequence[tuple[int, list[int]]],
    tests: Sequence[TestCase],
    expected_widths: Sequence[int] | None = None,
) -> set[int]:
    if not tests:
        raise ValueError("Task produced zero test cases")
    n_expected = len(tests[0].inputs)
    if n_expected == 0:
        raise ValueError("Task test cases declare zero input registers")
    for s, t in enumerate(tests):
        if len(t.inputs) != n_expected or len(t.expected) != n_expected:
            raise ValueError(
                f"Test case {s} register count mismatch: inputs={len(t.inputs)}, expected={len(t.expected)}, want={n_expected}"
            )
    if len(regs) == 0:
        raise ValueError("OPSTRM01 stream declares zero registers")
    if len(regs) != n_expected:
        raise ValueError(
            f"OPSTRM01 register count mismatch: stream declares {len(regs)}, task requires {n_expected}"
        )
    if expected_widths is not None:
        if len(expected_widths) != n_expected:
            raise ValueError(
                f"Task register_widths length ({len(expected_widths)}) != test register count ({n_expected})"
            )
        for idx, ((w, _), want_w) in enumerate(zip(regs, expected_widths)):
            if w != want_w:
                raise ValueError(
                    f"OPSTRM01 register {idx} width mismatch: stream declares {w}, task requires {want_w}"
                )
    init_wires: set[int] = set()
    for idx, (w, wires) in enumerate(regs):
        if w <= 0 or len(wires) != w:
            raise ValueError(f"OPSTRM01 register {idx} has invalid width {w} (wires={len(wires)})")
        max_val = 1 << w
        for wire in wires:
            if wire < 0:
                raise ValueError(f"OPSTRM01 register {idx} contains negative wire id {wire}")
            if wire in init_wires:
                raise ValueError(f"OPSTRM01 register {idx} duplicates wire id {wire}")
            init_wires.add(wire)
        for s, t in enumerate(tests):
            if not (0 <= t.inputs[idx] < max_val):
                raise ValueError(f"Test {s} input reg {idx} value {t.inputs[idx]} out of range for {w} bits")
            if not (0 <= t.expected[idx] < max_val):
                raise ValueError(f"Test {s} expected reg {idx} value {t.expected[idx]} out of range for {w} bits")
    return init_wires


def _iter_ops_from_fh(fh, n_ops: int, init_wires: set[int]):
    seen: set[int] = set()
    unpack_hdr = struct.Struct("<BB").unpack
    unpack_args = {
        0: lambda _: (),
        1: struct.Struct("<i").unpack,
        2: struct.Struct("<2i").unpack,
        3: struct.Struct("<3i").unpack,
        4: struct.Struct("<4i").unpack,
        5: struct.Struct("<5i").unpack,
    }
    for _ in range(n_ops):
        code, nargs = unpack_hdr(_read_exact(fh, 2))
        exp = EXPECTED_ARITY.get(code)
        if exp is None:
            raise ValueError(f"Unknown OPSTRM01 opcode {code}")
        if nargs != exp:
            raise ValueError(f"Invalid arity for opcode {code}: got {nargs}, expected {exp}")
        if nargs:
            args = unpack_args[nargs](_read_exact(fh, 4 * nargs))
            if code == _OP_QALLOC and args[0] in init_wires and args[0] not in seen:
                seen.add(args[0])
                continue
            yield (code, *args)
        else:
            yield (code,)
    if fh.read(1):
        raise ValueError("Trailing bytes after declared n_ops in OPSTRM01 stream")


class StandaloneKickmixEngine:
    """Self-contained streaming kickmix evaluator completely decoupled from ``flat_stream.Simulator``.

    Tracks one bit-packed trajectory vector per live wire, allocates independent SHAKE-256
    measurement coins per ``OP_HMR`` / ``OP_AND_REL`` event, enforces clean-ancilla and
    aliasing invariants, and accumulates randomized global/Z/CZ phase kickback across all shots.
    """

    def __init__(self, *, batch: int = 1, seed: int = 0):
        if batch <= 0:
            raise ValueError(f"batch must be positive, got {batch}")
        self.batch = batch
        self.seed = int(seed)
        self._seed_bytes = self.seed.to_bytes(8, "little", signed=True)
        self._mask_all = (1 << batch) - 1
        self._coin_nbytes = (batch + 7) // 8
        self._wires: dict[int, int] = {}
        self._active: set[int] = set()
        self._outcomes: dict[int, int] = {}
        self._meas_seq: int = 0
        self._phase_bits: int = 0
        self._peak_live: int = 0
        self.toffoli: int = 0
        self.clifford: int = 0

    def _next_coin_mask(self) -> int:
        seq = self._meas_seq
        self._meas_seq = seq + 1
        raw = hashlib.shake_256(self._seed_bytes + seq.to_bytes(8, "little")).digest(self._coin_nbytes)
        return int.from_bytes(raw, "little") & self._mask_all

    def write(self, wires: Sequence[int], vals: Sequence[int]) -> None:
        if len(vals) != self.batch:
            raise ValueError(f"Expected {self.batch} shot values, got {len(vals)}")
        limit = 1 << len(wires)
        for s, v in enumerate(vals):
            if v < 0 or v >= limit:
                raise ValueError(f"Value {v} out of range for {len(wires)}-bit register (shot {s})")
        for bit_idx, wid in enumerate(wires):
            if wid < 0:
                raise AssertionError(f"Negative wire id {wid}")
            packed = 0
            for s, v in enumerate(vals):
                if (v >> bit_idx) & 1:
                    packed |= 1 << s
            self._wires[wid] = packed
            self._active.add(wid)
        if len(self._active) > self._peak_live:
            self._peak_live = len(self._active)

    def read(self, wires: Sequence[int]) -> list[int]:
        out = [0] * self.batch
        for bit_idx, wid in enumerate(wires):
            if wid not in self._active:
                raise AssertionError(f"Output wire {wid} is not live at readout")
            bits = self._wires.get(wid, 0)
            while bits:
                lsb = bits & -bits
                out[lsb.bit_length() - 1] |= 1 << bit_idx
                bits ^= lsb
        return out

    def _exec_ops(self, ops) -> None:
        wires = self._wires
        active = self._active
        outcomes = self._outcomes
        mask_all = self._mask_all

        for op in ops:
            code = op[0]
            if code == _OP_QALLOC:
                w = op[1]
                if w < 0:
                    raise AssertionError(f"Negative wire id {w} in OP_QALLOC")
                if w in active:
                    raise AssertionError(f"Double-allocated qubit {w}")
                wires[w] = 0
                active.add(w)
                if len(active) > self._peak_live:
                    self._peak_live = len(active)
            elif code == _OP_QFREE:
                w = op[1]
                if w < 0:
                    raise AssertionError(f"Negative wire id {w} in OP_QFREE")
                if w not in active:
                    raise AssertionError(f"Freed dead/unallocated qubit {w}")
                if wires[w] != 0:
                    raise AssertionError(f"Recycled dirty qubit {w}: mask={wires[w]:#x}")
                active.remove(w)
                wires[w] = 0
            elif code == _OP_X:
                w = op[1]
                if w not in active:
                    raise AssertionError(f"OP_X on dead qubit {w}")
                wires[w] ^= mask_all
            elif code == _OP_CX:
                c, t = op[1], op[2]
                if c not in active or t not in active:
                    raise AssertionError(f"OP_CX on dead qubit ({c}, {t})")
                if c == t:
                    raise AssertionError(f"OP_CX aliased operand {c}")
                self.clifford += 1
                wires[t] ^= wires[c]
            elif code == _OP_SWAP:
                a, b = op[1], op[2]
                if a not in active or b not in active:
                    raise AssertionError(f"OP_SWAP on dead qubit ({a}, {b})")
                if a == b:
                    raise AssertionError(f"OP_SWAP aliased operand {a}")
                self.clifford += 1
                wires[a], wires[b] = wires[b], wires[a]
            elif code == _OP_CCX:
                c1, c2, t = op[1], op[2], op[3]
                if c1 not in active or c2 not in active or t not in active:
                    raise AssertionError(f"OP_CCX on dead qubit ({c1}, {c2}, {t})")
                if c1 == c2 or c1 == t or c2 == t:
                    raise AssertionError(f"OP_CCX aliased operands ({c1}, {c2}, {t})")
                self.toffoli += 1
                wires[t] ^= wires[c1] & wires[c2]
            elif code == _OP_CSWAP:
                c, a, b = op[1], op[2], op[3]
                if c not in active or a not in active or b not in active:
                    raise AssertionError(f"OP_CSWAP on dead qubit ({c}, {a}, {b})")
                if c == a or c == b or a == b:
                    raise AssertionError(f"OP_CSWAP aliased operands ({c}, {a}, {b})")
                self.toffoli += 1
                self.clifford += 2
                diff = (wires[a] ^ wires[b]) & wires[c]
                wires[a] ^= diff
                wires[b] ^= diff
            elif code == _OP_AND_COMP:
                a, b, t, pa, pb = op[1], op[2], op[3], op[4], op[5]
                if a not in active or b not in active or t not in active:
                    raise AssertionError(f"OP_AND_COMP on dead qubit ({a}, {b}, {t})")
                if a == b or a == t or b == t:
                    raise AssertionError(f"OP_AND_COMP aliased operands ({a}, {b}, {t})")
                if wires[t] != 0:
                    raise AssertionError(f"and_compute target {t} not |0>")
                self.toffoli += 1
                va = wires[a] if pa else (wires[a] ^ mask_all)
                vb = wires[b] if pb else (wires[b] ^ mask_all)
                wires[t] = va & vb
            elif code == _OP_AND_REL:
                a, b, t, pa, pb = op[1], op[2], op[3], op[4], op[5]
                if a not in active or b not in active or t not in active:
                    raise AssertionError(f"OP_AND_REL on dead qubit ({a}, {b}, {t})")
                if a == b or a == t or b == t:
                    raise AssertionError(f"OP_AND_REL aliased operands ({a}, {b}, {t})")
                self.clifford += 2
                coins = self._next_coin_mask()
                va = wires[a] if pa else (wires[a] ^ mask_all)
                vb = wires[b] if pb else (wires[b] ^ mask_all)
                self._phase_bits ^= coins & (wires[t] ^ (va & vb))
                active.remove(t)
                wires[t] = 0
            elif code == _OP_HMR:
                w, oid = op[1], op[2]
                if w not in active:
                    raise AssertionError(f"OP_HMR on dead qubit {w}")
                self.clifford += 1
                coins = self._next_coin_mask()
                outcomes[oid] = coins
                self._phase_bits ^= coins & wires[w]
                wires[w] = 0
            elif code == _OP_OCLEAR:
                outcomes[op[1]] = 0
            elif code == _OP_OXOR:
                dst, src = op[1], op[2]
                if src not in outcomes:
                    raise AssertionError(f"OP_OXOR references undefined source outcome id {src}")
                if dst not in outcomes:
                    raise AssertionError(f"OP_OXOR references undefined destination outcome id {dst}")
                outcomes[dst] ^= outcomes[src]
            elif code == _OP_NEG:
                cid = op[1]
                if cid >= 0 and cid not in outcomes:
                    raise AssertionError(f"OP_NEG references undefined outcome id {cid}")
                self._phase_bits ^= mask_all if cid < 0 else outcomes[cid]
            elif code == _OP_Z:
                w, cid = op[1], op[2]
                if w not in active:
                    raise AssertionError(f"OP_Z on dead qubit {w}")
                if cid >= 0 and cid not in outcomes:
                    raise AssertionError(f"OP_Z references undefined outcome id {cid}")
                cmask = mask_all if cid < 0 else outcomes[cid]
                self._phase_bits ^= cmask & wires[w]
            elif code == _OP_CZ:
                a, b, cid = op[1], op[2], op[3]
                if a not in active or b not in active:
                    raise AssertionError(f"OP_CZ on dead qubit ({a}, {b})")
                if a == b:
                    raise AssertionError(f"OP_CZ aliased operand {a}")
                if cid >= 0 and cid not in outcomes:
                    raise AssertionError(f"OP_CZ references undefined outcome id {cid}")
                self.clifford += 1
                cmask = mask_all if cid < 0 else outcomes[cid]
                self._phase_bits ^= cmask & wires[a] & wires[b]
            else:
                raise ValueError(f"Unknown OPSTRM01 opcode {code}")

    def assert_clean(self, outputs: Sequence[int], phase: int = 0) -> None:
        out_set = set(outputs)
        missing = [w for w in outputs if w not in self._active]
        if missing:
            raise AssertionError(f"Missing freed output wires: {missing[:10]}")
        leaked = [w for w in self._active if w not in out_set]
        if leaked:
            raise AssertionError(f"Leaked live ancilla wires: {leaked[:10]}")
        dirty = [w for w, val in self._wires.items() if w not in out_set and val != 0]
        if dirty:
            raise AssertionError(f"Dirty non-output wires: {dirty[:10]}")
        target_p = self._mask_all if phase else 0
        if self._phase_bits != target_p:
            raise AssertionError(f"Uncancelled randomized phase mask={self._phase_bits:#x}")


class _CompactVM:
    def __init__(
        self,
        fh,
        tests: Sequence[TestCase],
        sim=None,
        expected_widths: Sequence[int] | None = None,
        seed: int = 0,
    ):
        # Caller already consumed b"OPSTRM01"
        self.registers, self._n_ops = _read_header(fh)
        self._init_wires = _validate_register_schema(self.registers, tests, expected_widths=expected_widths)
        self._fh = fh
        self.tests = tests
        self.sim = StandaloneKickmixEngine(batch=len(tests), seed=seed) if sim is None else sim

    def run(self) -> None:
        for idx, (_, wires) in enumerate(self.registers):
            self.sim.write(wires, [t.inputs[idx] for t in self.tests])
        self.sim._exec_ops(_iter_ops_from_fh(self._fh, self._n_ops, self._init_wires))

    def flush(self) -> None:
        pass


def run_task(task: TaskProtocol, workdir: str, fh, h_ops: bytes, jobs: int = 1, stream_path: str | None = None):
    try:
        hdr = _read_exact(fh, 8)
        if hdr != b"OPSTRM01":
            return 1, {"verdict": "FAIL", "reason": f"Bad stream magic: {hdr!r}"}
        regs, n_ops = _read_header(fh)
        tests = [task.gen_test_indexed(s, random.Random(0x5EED_0000 ^ s)) for s in range(task.num_tests)]
        expected_widths = _extract_expected_widths(task)
        init_wires = _validate_register_schema(regs, tests, expected_widths=expected_widths)
        seed_int = int.from_bytes(hashlib.sha256(task.seed_domain + h_ops).digest()[:8], "little", signed=True)
        engine = StandaloneKickmixEngine(batch=len(tests), seed=seed_int)
        for idx, (_, wires) in enumerate(regs):
            engine.write(wires, [t.inputs[idx] for t in tests])
        engine._exec_ops(_iter_ops_from_fh(fh, n_ops, init_wires))
    except (AssertionError, ValueError) as exc:
        return 1, {"verdict": "FAIL", "reason": str(exc)}
    try:
        for idx, (_, wires) in enumerate(regs):
            got = engine.read(wires)
            exp = [t.expected[idx] for t in tests]
            if got != exp:
                return 1, {"verdict": "FAIL", "reason": f"reg {idx} mismatch"}
        engine.assert_clean(outputs=[w for _, wires in regs for w in wires], phase=0)
    except AssertionError as exc:
        return 1, {"verdict": "FAIL", "reason": str(exc)}
    return 0, {
        "verdict": "OK",
        "metrics": {
            "qubits": engine._peak_live,
            "toffoli": float(engine.toffoli),
            "clifford": float(engine.clifford),
        },
    }
