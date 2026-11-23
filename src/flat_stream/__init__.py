"""Independent flat op-stream Builder & strict all-2^M symbolic phase Simulator
clean-room implementation compatible with the OPSTRM01 wire-stream interface of Gu et al. (2026) (https://github.com/andigu/uncompute, Apache-2.0) and randomized kickmix (Khattar et al., 2026).
"""
from __future__ import annotations

import hashlib
import struct
from typing import Sequence

OP_X = 0
OP_Z = 1
OP_NEG = 2
OP_CX = 3
OP_CZ = 4
OP_SWAP = 5
OP_CCX = 6
OP_HMR = 9
OP_OXOR = 10
OP_AND_COMP = 20
OP_AND_REL = 21
OP_QALLOC = 22
OP_QFREE = 23
OP_OCLEAR = 24
OP_CSWAP = 25


class Outcome:
    __slots__ = ("_b", "id")

    def __init__(self, builder: "Builder", oid: int):
        self._b = builder
        self.id = oid

    def __xor__(self, other: "Outcome") -> "Outcome":
        dst = self._b._alloc_out()
        self._b._emit_op((OP_OCLEAR, dst))
        self._b._emit_op((OP_OXOR, dst, self.id))
        self._b._emit_op((OP_OXOR, dst, other.id))
        return Outcome(self._b, dst)

    def __ixor__(self, other: "Outcome") -> "Outcome":
        self._b._emit_op((OP_OXOR, self.id, other.id))
        return self


class Builder:
    def __init__(self, recycle_outcomes: bool = True):
        self.recycle_outcomes = recycle_outcomes
        self._free_q: list[int] = []
        self._next_q: int = 0
        self._live_q: int = 0
        self.peak_qubits: int = 0
        self._free_o: list[int] = []
        self.next_out: int = 0
        self.registers: list[list[int]] = []
        self.ops: list[tuple] = []
        self._phase_name: str = "default"
        self._phase_order: list[str] = []
        self._phase_stats: dict[str, dict[str, int]] = {}
        self._cur_stats: dict[str, int] = self._ensure_phase("default")

    def _ensure_phase(self, name: str) -> dict[str, int]:
        if name not in self._phase_stats:
            self._phase_order.append(name)
            self._phase_stats[name] = {"toffoli": 0, "clifford": 0}
        return self._phase_stats[name]

    def set_phase(self, name: str) -> None:
        self._phase_name = name
        self._cur_stats = self._ensure_phase(name)

    def phase_summary(self) -> dict:
        phases = [
            {"name": n, "toffoli": self._phase_stats[n]["toffoli"], "clifford": self._phase_stats[n]["clifford"]}
            for n in self._phase_order
            if self._phase_stats[n]["toffoli"] or self._phase_stats[n]["clifford"] or n in ("loop1", "prime_kernel", "loop1_final", "final_phase")
        ]
        tot_t = sum(v["toffoli"] for v in self._phase_stats.values())
        tot_c = sum(v["clifford"] for v in self._phase_stats.values())
        return {
            "phases": phases,
            "total_emitted_toffoli": tot_t,
            "total_emitted_clifford": tot_c,
            "peak_qubits": self.peak_qubits,
        }

    def _emit_op(self, op: tuple) -> None:
        self.ops.append(op)

    def _alloc_out(self) -> int:
        if self.recycle_outcomes and self._free_o:
            return self._free_o.pop()
        oid = self.next_out
        self.next_out += 1
        return oid

    def retire(self, m: Outcome) -> None:
        if self.recycle_outcomes:
            self._free_o.append(m.id)

    def qubit(self) -> int:
        if self._free_q:
            q = self._free_q.pop()
        else:
            q = self._next_q
            self._next_q += 1
        self._live_q += 1
        if self._live_q > self.peak_qubits:
            self.peak_qubits = self._live_q
        self._emit_op((OP_QALLOC, q))
        return q

    def qubits(self, n: int) -> list[int]:
        return [self.qubit() for _ in range(n)]

    def recycle(self, q: int) -> None:
        self._live_q -= 1
        self._free_q.append(q)
        self._emit_op((OP_QFREE, q))

    def qubit_register(self, wires: Sequence[int]) -> None:
        self.registers.append(list(wires))

    def x(self, q: int) -> None:
        self._emit_op((OP_X, q))

    def cx(self, c: int, t: int) -> None:
        self._cur_stats["clifford"] += 1
        self._emit_op((OP_CX, c, t))

    def swap(self, a: int, b: int) -> None:
        self._cur_stats["clifford"] += 1
        self._emit_op((OP_SWAP, a, b))

    def ccx(self, c1: int, c2: int, t: int) -> None:
        self._cur_stats["toffoli"] += 1
        self._emit_op((OP_CCX, c1, c2, t))

    def cswap(self, c: int, a: int, b: int) -> None:
        self._cur_stats["toffoli"] += 1
        self._cur_stats["clifford"] += 2
        self._emit_op((OP_CSWAP, c, a, b))

    def z(self, q: int, cond: Outcome | None = None) -> None:
        self._emit_op((OP_Z, q, -1 if cond is None else cond.id))

    def cz(self, a: int, b: int, cond: Outcome | None = None) -> None:
        self._cur_stats["clifford"] += 1
        self._emit_op((OP_CZ, a, b, -1 if cond is None else cond.id))

    def neg(self, cond: Outcome | None = None) -> None:
        self._emit_op((OP_NEG, -1 if cond is None else cond.id))

    def hmr(self, q: int) -> Outcome:
        oid = self._alloc_out()
        self._cur_stats["clifford"] += 1
        self._emit_op((OP_HMR, q, oid))
        return Outcome(self, oid)

    def and_compute(self, a: int, b: int, pol: tuple[bool, bool] = (True, True)) -> int:
        t = self.qubit()
        self._cur_stats["toffoli"] += 1
        self._emit_op((OP_AND_COMP, a, b, t, 1 if pol[0] else 0, 1 if pol[1] else 0))
        return t

    def and_release(self, t: int, a: int, b: int, pol: tuple[bool, bool] = (True, True)) -> None:
        self._cur_stats["clifford"] += 2  # HMR + conditional CZ
        self._live_q -= 1
        self._free_q.append(t)
        self._emit_op((OP_AND_REL, a, b, t, 1 if pol[0] else 0, 1 if pol[1] else 0))

    def emit(self, meta: bool = False, dest=None) -> None:
        assert dest is not None
        dest.write(b"OPSTRM01")
        dest.write(struct.pack("<II", len(self.registers), len(self.ops)))
        for reg in self.registers:
            dest.write(struct.pack("<I", len(reg)))
            if reg:
                dest.write(struct.pack(f"<{len(reg)}I", *reg))
        # Serialize ops compactly for independent replay + SHA-256
        for op in self.ops:
            code = op[0]
            args = op[1:]
            dest.write(struct.pack("<BB", code, len(args)))
            for a in args:
                dest.write(struct.pack("<i", int(a)))


class Simulator:
    """Strict all-2^M symbolic measurement-outcome phase & clean-ancilla verifier."""

    def __init__(self, builder: Builder | None = None, *, batch: int = 1, strict: bool = True, seed: int = 0):
        self.builder = builder
        self.batch = batch
        self.strict = strict
        self.seed = seed
        self._all_mask = (1 << batch) - 1
        self._coin_bytes = (batch + 7) // 8
        # Each qubit state is an int bitmask across `batch` parallel shots!
        self._q: dict[int, int] = {}
        self._live: set[int] = set()
        # Symbolic mode: out_sym[oid] is a set[int] of primitive HMR indices (input-independent).
        # Randomized mode (strict=False): out_sym[oid] is a `batch`-bit int bitmask with one independent coin per shot.
        self._out_sym: dict[int, object] = {}
        self._hmr_count: int = 0
        # phase_poly[shot] is a set[int]: 0 = constant (-1), (k+1) = HMR_k parity (exact F2 vector)
        self._phase_poly: list[set[int]] = [set() for _ in range(batch)]
        self._peak_live: int = 0
        self.toffoli: int = 0
        self.clifford: int = 0
        self._phase_rand: int = 0

    def _draw_shot_coins(self, k: int) -> int:
        raw = hashlib.shake_256(
            self.seed.to_bytes(8, "little", signed=True) + k.to_bytes(8, "little")
        ).digest(self._coin_bytes)
        return int.from_bytes(raw, "little") & self._all_mask

    def write(self, wires: Sequence[int], vals: Sequence[int]) -> None:
        assert len(vals) == self.batch
        max_val = 1 << len(wires)
        for s, v in enumerate(vals):
            if v < 0 or v >= max_val:
                raise ValueError(f"Value {v} out of range for {len(wires)}-bit register (shot {s})")
        for bit_i, w in enumerate(wires):
            if w < 0:
                raise AssertionError(f"Negative wire id {w}")
            mask = 0
            for s, v in enumerate(vals):
                if (v >> bit_i) & 1:
                    mask |= (1 << s)
            self._q[w] = mask
            self._live.add(w)
        if len(self._live) > self._peak_live:
            self._peak_live = len(self._live)

    def read(self, wires: Sequence[int]) -> list[int]:
        out = [0] * self.batch
        for bit_i, w in enumerate(wires):
            if w not in self._live:
                raise AssertionError(f"Output wire {w} is not live at readout")
            qmask = self._q.get(w, 0)
            if qmask:
                m = qmask
                while m:
                    lsb = m & -m
                    s = lsb.bit_length() - 1
                    out[s] |= (1 << bit_i)
                    m ^= lsb
        return out

    def _xor_phase_mask(self, active_shots_mask: int, sym_set: set[int]) -> None:
        if not active_shots_mask or not sym_set:
            return
        m = active_shots_mask
        polys = self._phase_poly
        while m:
            lsb = m & -m
            s = lsb.bit_length() - 1
            polys[s].symmetric_difference_update(sym_set)
            m ^= lsb

    def _exec_ops(self, ops: Sequence[tuple]) -> None:
        all_m = self._all_mask
        q = self._q
        live = self._live
        out_sym = self._out_sym
        strict = self.strict

        for op in ops:
            code = op[0]
            if code == OP_QALLOC:
                w = op[1]
                if w < 0:
                    raise AssertionError(f"Negative wire id {w} in OP_QALLOC")
                if w in live:
                    raise AssertionError(f"Double-allocated qubit {w}")
                q[w] = 0
                live.add(w)
                if len(live) > self._peak_live:
                    self._peak_live = len(live)
            elif code == OP_QFREE:
                w = op[1]
                if w < 0:
                    raise AssertionError(f"Negative wire id {w} in OP_QFREE")
                if w not in live:
                    raise AssertionError(f"Freed dead/unallocated qubit {w}")
                if q.get(w, 0) != 0:
                    raise AssertionError(f"Recycled dirty qubit {w}: mask={q.get(w,0):#x}")
                live.discard(w)
                q[w] = 0
            elif code == OP_X:
                w = op[1]
                if w not in live:
                    raise AssertionError(f"OP_X on dead qubit {w}")
                q[w] = q.get(w, 0) ^ all_m
            elif code == OP_CX:
                c, t = op[1], op[2]
                if c not in live or t not in live:
                    raise AssertionError(f"OP_CX on dead qubit ({c}, {t})")
                if c == t:
                    raise AssertionError(f"OP_CX aliased operand {c}")
                self.clifford += 1
                q[t] = q.get(t, 0) ^ q.get(c, 0)
            elif code == OP_SWAP:
                a, b = op[1], op[2]
                if a not in live or b not in live:
                    raise AssertionError(f"OP_SWAP on dead qubit ({a}, {b})")
                if a == b:
                    raise AssertionError(f"OP_SWAP aliased operand {a}")
                self.clifford += 1
                qa, qb = q.get(a, 0), q.get(b, 0)
                q[a], q[b] = qb, qa
            elif code == OP_CCX:
                c1, c2, t = op[1], op[2], op[3]
                if c1 not in live or c2 not in live or t not in live:
                    raise AssertionError(f"OP_CCX on dead qubit ({c1}, {c2}, {t})")
                if c1 == c2 or c1 == t or c2 == t:
                    raise AssertionError(f"OP_CCX aliased operands ({c1}, {c2}, {t})")
                self.toffoli += 1
                q[t] = q.get(t, 0) ^ (q.get(c1, 0) & q.get(c2, 0))
            elif code == OP_CSWAP:
                c, a, b = op[1], op[2], op[3]
                if c not in live or a not in live or b not in live:
                    raise AssertionError(f"OP_CSWAP on dead qubit ({c}, {a}, {b})")
                if c == a or c == b or a == b:
                    raise AssertionError(f"OP_CSWAP aliased operands ({c}, {a}, {b})")
                self.toffoli += 1
                self.clifford += 2
                cm = q.get(c, 0)
                diff = (q.get(a, 0) ^ q.get(b, 0)) & cm
                q[a] = q.get(a, 0) ^ diff
                q[b] = q.get(b, 0) ^ diff
            elif code == OP_AND_COMP:
                a, b, t, pa, pb = op[1], op[2], op[3], op[4], op[5]
                if a not in live or b not in live or t not in live:
                    raise AssertionError(f"OP_AND_COMP on dead qubit ({a}, {b}, {t})")
                if a == b or a == t or b == t:
                    raise AssertionError(f"OP_AND_COMP aliased operands ({a}, {b}, {t})")
                if q.get(t, 0) != 0:
                    raise AssertionError(f"and_compute target {t} not |0>")
                self.toffoli += 1
                qa = q.get(a, 0) if pa else (q.get(a, 0) ^ all_m)
                qb = q.get(b, 0) if pb else (q.get(b, 0) ^ all_m)
                q[t] = qa & qb
            elif code == OP_AND_REL:
                a, b, t, pa, pb = op[1], op[2], op[3], op[4], op[5]
                if a not in live or b not in live or t not in live:
                    raise AssertionError(f"OP_AND_REL on dead qubit ({a}, {b}, {t})")
                if a == b or a == t or b == t:
                    raise AssertionError(f"OP_AND_REL aliased operands ({a}, {b}, {t})")
                self.clifford += 2
                # Explicitly simulate HMR(t) measurement phase kickback + polarity-adjusted conditional CZ(a, b)
                k = self._hmr_count
                self._hmr_count += 1
                qt = q.get(t, 0)
                qa = q.get(a, 0) if pa else (q.get(a, 0) ^ all_m)
                qb = q.get(b, 0) if pb else (q.get(b, 0) ^ all_m)
                ab = qa & qb
                if strict:
                    sym = {k + 1}
                    if qt:
                        self._xor_phase_mask(qt, sym)
                    if ab:
                        self._xor_phase_mask(ab, sym)
                else:
                    cmask = self._draw_shot_coins(k)
                    if qt:
                        self._phase_rand ^= (cmask & qt)
                    if ab:
                        self._phase_rand ^= (cmask & ab)
                live.discard(t)
                q[t] = 0
            elif code == OP_HMR:
                w, oid = op[1], op[2]
                if w not in live:
                    raise AssertionError(f"OP_HMR on dead qubit {w}")
                self.clifford += 1
                k = self._hmr_count
                self._hmr_count += 1
                qw = q.get(w, 0)
                if strict:
                    sym = {k + 1}
                    out_sym[oid] = sym
                    if qw:
                        self._xor_phase_mask(qw, sym)
                else:
                    # Independent per-shot randomized measurement bits via SHAKE-256
                    cmask = self._draw_shot_coins(k)
                    out_sym[oid] = cmask
                    if qw:
                        self._phase_rand ^= (cmask & qw)
                q[w] = 0
            elif code == OP_OCLEAR:
                out_sym[op[1]] = set() if strict else 0
            elif code == OP_OXOR:
                dst, src = op[1], op[2]
                if src not in out_sym:
                    raise AssertionError(f"OP_OXOR references undefined source outcome id {src}")
                if dst not in out_sym:
                    raise AssertionError(f"OP_OXOR references undefined destination outcome id {dst}")
                if strict:
                    ss = out_sym[src]
                    if ss:
                        out_sym[dst].symmetric_difference_update(ss)
                else:
                    out_sym[dst] = out_sym[dst] ^ out_sym[src]
            elif code == OP_NEG:
                cid = op[1]
                if cid >= 0 and cid not in out_sym:
                    raise AssertionError(f"OP_NEG references undefined outcome id {cid}")
                if strict:
                    sym = {0} if cid < 0 else out_sym[cid]
                    if sym:
                        for p in self._phase_poly:
                            p.symmetric_difference_update(sym)
                else:
                    cmask = all_m if cid < 0 else out_sym[cid]
                    if cmask:
                        self._phase_rand ^= cmask
            elif code == OP_Z:
                w, cid = op[1], op[2]
                if w not in live:
                    raise AssertionError(f"OP_Z on dead qubit {w}")
                if cid >= 0 and cid not in out_sym:
                    raise AssertionError(f"OP_Z references undefined outcome id {cid}")
                qw = q.get(w, 0)
                if qw:
                    if strict:
                        sym = {0} if cid < 0 else out_sym[cid]
                        if sym:
                            self._xor_phase_mask(qw, sym)
                    else:
                        cmask = all_m if cid < 0 else out_sym[cid]
                        if cmask:
                            self._phase_rand ^= (cmask & qw)
            elif code == OP_CZ:
                a, b, cid = op[1], op[2], op[3]
                if a not in live or b not in live:
                    raise AssertionError(f"OP_CZ on dead qubit ({a}, {b})")
                if a == b:
                    raise AssertionError(f"OP_CZ aliased operand {a}")
                if cid >= 0 and cid not in out_sym:
                    raise AssertionError(f"OP_CZ references undefined outcome id {cid}")
                self.clifford += 1
                ab = q.get(a, 0) & q.get(b, 0)
                if ab:
                    if strict:
                        sym = {0} if cid < 0 else out_sym[cid]
                        if sym:
                            self._xor_phase_mask(ab, sym)
                    else:
                        cmask = all_m if cid < 0 else out_sym[cid]
                        if cmask:
                            self._phase_rand ^= (cmask & ab)
            else:
                raise ValueError(f"Unknown op code {code}")

    def run(self) -> None:
        assert self.builder is not None
        # Ensure input registers are marked live
        for reg in self.builder.registers:
            for w in reg:
                if w < 0:
                    raise AssertionError(f"Negative wire id {w} in register")
                self._live.add(w)
        if len(self._live) > self._peak_live:
            self._peak_live = len(self._live)
        # Skip initial QALLOCs for input registers if already written
        init_wires = {w for reg in self.builder.registers for w in reg}
        seen_init = set()
        def _gen():
            for op in self.builder.ops:
                if op[0] == OP_QALLOC and op[1] in init_wires and op[1] not in seen_init:
                    seen_init.add(op[1])
                    continue
                yield op
        self._exec_ops(_gen())

    def assert_clean(self, outputs: Sequence[int], phase: int = 0) -> None:
        out_set = set(outputs)
        missing = [w for w in outputs if w not in self._live]
        if missing:
            raise AssertionError(f"Missing freed output wires: {missing[:10]}")
        leaked = [w for w in self._live if w not in out_set]
        if leaked:
            raise AssertionError(f"Leaked live ancilla wires: {leaked[:10]}")
        dirty = [w for w, m in self._q.items() if w not in out_set and m != 0]
        if dirty:
            raise AssertionError(f"Dirty non-output wires: {dirty[:10]}")
        if self.strict:
            target_p = {0} if phase else set()
            for s, p in enumerate(self._phase_poly):
                if p != target_p:
                    raise AssertionError(
                        f"Uncancelled symbolic phase on shot {s}: len={len(p)}, sample={sorted(p)[:8]} (expected {target_p})"
                    )
        else:
            target_m = self._all_mask if phase else 0
            if self._phase_rand != target_m:
                raise AssertionError(f"Uncancelled randomized phase mask={self._phase_rand:#x}")
