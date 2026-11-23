"""Verified Gate-Level Primitives for the RSA-2048 Batched RNS + Dyadic Torus Flat Compiler.

Key verified (`kickmix` + `Simulator(strict=True)` symbolic-ledger) mechanisms:
1. ``compute_all_crt_pairs``: Exact ``O(P log^2 P)`` binary product/remainder tree computing
   ``((L/p_i)^-1 mod p_i, (L/p_i) mod N)`` for ~20,000 primes in <0.5s with zero 400k-bit divisions.
2. ``emit_fast_nonrestoring_loop2``: Non-restoring ``ctz(p-1)``-sliced modular division/uncomputation
   using 1 uncontrolled conjugated constant adder per middle bit instead of 2 restoring adders.
3. ``emit_bilinear_anf_add_zero_phase``: Bilinear ``low_map x high_map`` shared-selector ANF QROM +
   addition/subtraction + zero-Toffoli ``CZ(low_map, high_map, cond=...)`` X-measurement cleanup,
   using only ``(2^sb - sb - 1) + (2^ab - ab - 1) + 1`` ancilla wires (~6-9 qubits instead of 26-57!).
4. ``emit_rolling_anf_loop1_update``: Zero-phase, zero-boundary-tax Rolling-ANF Loop 1 update
   (full materialized ANF for ``k >= 2``, or bilinear ``3b x 3b`` zero-phase streaming for ``k == 1``),
   including the ``mask == 0`` ``b.neg(cond=...)`` global phase correction.
5. ``emit_universal_cz_phaseup`` / ``emit_asymmetric_half_dfs_phaseup``: Static outcome-independent
   All-to-All CZ DAG phase synthesis for ``n = 5..8`` (``r_n = 4, 7, 11, 17`` Toffolis) and
   configurable ``(n1, n2)`` Half-DFS for ``n >= 9`` (including Q-safe ``(6, 10)`` at ``a = 16`` for ``k = 1``).
"""
from __future__ import annotations

from typing import Sequence
import numpy as np

from flat_stream.gadgets.arith import add as _gu_add
from flat_stream.gadgets.qrom import swap_perm, walk, xor_word as _py_xor_word


# Frozen verified Universal All-to-All CZ AND-DAG recipes for n = 5, 6, 7, 8:
# K = 4 (vs Half-DFS 5), 7 (vs 8), 11 (vs 15), 17 (vs 22).
FROZEN_UNIVERSAL_CZ_RECIPES: dict[int, list[tuple[list[int], list[int]]]] = {
    5: [([3, 0], [3, 2, 1]), ([3], [1]), ([6, 4], [0, 1]), ([2, 1], [4, 0])],
    6: [
        ([2, 0], [3, 2]), ([3], [1]), ([7, 0, 3], [2, 4]), ([1, 0, 3], [6, 5]),
        ([9, 1, 8], [1, 4]), ([5, 3], [7, 3, 4]), ([9, 11, 2], [3, 8]),
    ],
    7: [
        ([4], [6, 0]), ([5, 7, 4], [2, 4]), ([6], [4, 1]), ([7, 8], [3, 7, 9]),
        ([7, 9, 10], [0, 8]), ([1], [2]), ([11, 10, 5], [0, 9, 4]),
        ([9, 13, 3], [11, 4, 8]), ([1, 6, 0], [3]), ([3, 8], [3, 0, 5]), ([7, 2, 16], [0, 3]),
    ],
    8: [
        ([2, 4], [1, 3, 7]), ([6, 3], [1, 5]), ([5, 3, 4], [4, 1, 2]), ([8, 10], [4]),
        ([3], [4, 0, 7]), ([9, 7, 3], [2, 11, 5]), ([10, 3, 0], [1, 11, 7]),
        ([9, 5, 0], [2, 5, 6]), ([13, 14], [8, 6]), ([0, 7, 1], [16, 15, 12]),
        ([17, 0, 7], [10, 3]), ([2, 11], [0, 17]), ([16, 6, 12], [4, 8, 18]),
        ([1, 17, 11], [9, 7, 12]), ([2, 6], [9, 2]), ([15, 9, 1], [14, 18]), ([19, 10, 4], [22]),
    ],
}


def primitive_root_prime(p: int) -> int:
    """Deterministic smallest positive primitive root modulo prime ``p`` (matches ``sympy.primitive_root(p)``)."""
    if p == 2:
        return 1
    phi = p - 1
    n = phi
    factors: list[int] = []
    if n % 2 == 0:
        factors.append(2)
        while n % 2 == 0:
            n //= 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 2
    if n > 1:
        factors.append(n)
    exps = [phi // q for q in factors]
    g = 2
    while g < p:
        if all(pow(g, e, p) != 1 for e in exps):
            return g
        g += 1
    raise ValueError(f"No primitive root found for {p}")


def is_prime_u32(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3, 5, 7):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = n - 1
    s = (d & -d).bit_length() - 1
    d >>= s
    for a in (2, 7, 61):
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def compute_all_crt_pairs(periods: Sequence[int], modulus: int) -> tuple[tuple[int, int], ...]:
    """Compute ``((L/p_i)^-1 mod p_i, (L/p_i) mod modulus)`` for all ``p_i`` in ``periods``
    via prefix/suffix products mod ``modulus`` + a binary product/remainder tree.
    """
    m = len(periods)
    if m == 0:
        return ()
    if m == 1:
        return ((1 % periods[0], 1 % modulus),)

    # 1. Prefix-suffix products modulo `modulus` -> O(m) 2048-bit multiplies!
    pref = [1] * m
    acc = 1
    for i in range(m):
        pref[i] = acc
        acc = (acc * periods[i]) % modulus
    q_mod_n = [0] * m
    acc = 1
    for i in range(m - 1, -1, -1):
        q_mod_n[i] = (pref[i] * acc) % modulus
        acc = (acc * periods[i]) % modulus

    # 2. Binary product tree + top-down remainder propagation for `(L / p_i) % p_i`!
    levels: list[list[int]] = [list(map(int, periods))]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        nxt = []
        for i in range(0, len(cur), 2):
            if i + 1 < len(cur):
                nxt.append(cur[i] * cur[i + 1])
            else:
                nxt.append(cur[i])
        levels.append(nxt)

    rem_cur = [1 % levels[-1][0]]
    for lvl in range(len(levels) - 2, -1, -1):
        child_prods = levels[lvl]
        rem_nxt = [0] * len(child_prods)
        for p_idx, r_val in enumerate(rem_cur):
            li = 2 * p_idx
            ri = li + 1
            if ri < len(child_prods):
                lp = child_prods[li]
                rp = child_prods[ri]
                rem_nxt[li] = ((r_val % lp) * (rp % lp)) % lp
                rem_nxt[ri] = ((r_val % rp) * (lp % rp)) % rp
            else:
                lp = child_prods[li]
                rem_nxt[li] = r_val % lp
        rem_cur = rem_nxt

    return tuple((pow(rem_cur[i], -1, periods[i]), q_mod_n[i]) for i in range(m))


def dyadic_crt_entry_value(term: int, p: int, inv_q: int, q_mod_n: int, modulus: int, len_acc: int) -> int:
    """Exact rounded dyadic fixed-point CRT entry ``round(((q_i * k) mod N) * 2^f / N) mod 2^f``."""
    t_mod = term % p
    if t_mod == 0:
        return 0
    k = (inv_q * t_mod) % p
    contrib = (q_mod_n * k) % modulus
    scaled = ((contrib << len_acc) + (modulus >> 1)) // modulus
    return scaled & ((1 << len_acc) - 1)


def xor_word(b, ctrl, word: int, target: Sequence[int]) -> None:
    if not word:
        return
    _py_xor_word(b, ctrl, word, target)


def fast_fanout_anf_lane(b, m2w: dict[int, int | None], anf_lane: Sequence[int], word_reg: Sequence[int]) -> None:
    """Fan out ``anf_lane[0..2^n-1]`` into ``word_reg``."""
    for mask, wv in enumerate(anf_lane):
        if wv:
            xor_word(b, m2w[mask], wv, word_reg)


def _fast_ixor_outcomes(b, dst_outcome, handles: list) -> None:
    for h in handles:
        dst_outcome ^= h


def ctz(val: int) -> int:
    if val <= 0:
        raise ValueError("ctz requires a positive integer")
    return (val & -val).bit_length() - 1


def mobius_transform_u64_matrix(mat: np.ndarray, n_bits: int) -> np.ndarray:
    out = np.array(mat, dtype=np.uint64, copy=True)
    if out.shape[0] != (1 << n_bits):
        raise ValueError(f"Expected {1 << n_bits} rows, got {out.shape[0]}")
    for bit in range(n_bits):
        step = 1 << bit
        view = out.reshape(-1, 2, step, out.shape[1])
        view[:, 1, :, :] ^= view[:, 0, :, :]
    return out


def mobius_transform_int_list(values: Sequence[int], n_bits: int) -> list[int]:
    out = [int(v) for v in values]
    if len(out) != (1 << n_bits):
        raise ValueError(f"Expected {1 << n_bits} entries, got {len(out)}")
    for bit in range(n_bits):
        step = 1 << bit
        span = step << 1
        for base in range(0, 1 << n_bits, span):
            for off in range(step):
                out[base + step + off] ^= out[base + off]
    return out


def words_from_int_masks(masks: Sequence[int], n_outcomes: int) -> np.ndarray:
    """Pack length-``2^n`` integer bitmasks over ``n_outcomes`` into ``(2^n, n_words)`` uint64."""
    n_rows = len(masks)
    n_words = max(1, (n_outcomes + 63) // 64)
    out = np.zeros((n_rows, n_words), dtype=np.uint64)
    for w_i in range(n_words):
        shift = w_i * 64
        out[:, w_i] = np.fromiter(((int(v) >> shift) & 0xFFFFFFFFFFFFFFFF for v in masks), dtype=np.uint64, count=n_rows)
    return out


class NibbleOutcomeSynthesizer:
    """Bounded-ID 4-bit nibble or 8-bit byte synthesizer for symbolic ``Outcome`` parities."""

    def __init__(self, b, base_outcomes: Sequence, *, use_bytes: bool = False):
        self.b = b
        self.base = list(base_outcomes)
        self.k = len(self.base)
        self.use_bytes = use_bytes and self.k >= 8
        self._roll = None
        self._roll_words: np.ndarray | None = None
        self._roll_int: int = 0
        self._derived: list = []

        if self.use_bytes:
            self.byte_tables: list[list[object | None]] = []
            for start in range(0, self.k, 8):
                chunk = self.base[start : start + 8]
                m = len(chunk)
                tab: list[object | None] = [None] * (1 << m)
                for bit in range(m):
                    tab[1 << bit] = chunk[bit]
                for mask in range(3, 1 << m):
                    if (mask & (mask - 1)) == 0:
                        continue
                    lsb = mask & -mask
                    derived = tab[mask ^ lsb] ^ tab[lsb]
                    tab[mask] = derived
                    self._derived.append(derived)
                if m < 8:
                    tab.extend([None] * (256 - len(tab)))
                self.byte_tables.append(tab)
            self._n_bt = len(self.byte_tables)
        else:
            self.nibbles: list[dict[int, object]] = []
            for start in range(0, self.k, 4):
                chunk = self.base[start : start + 4]
                m = len(chunk)
                table: dict[int, object] = {}
                for bit in range(m):
                    table[1 << bit] = chunk[bit]
                for mask in range(3, 1 << m):
                    if (mask & (mask - 1)) == 0:
                        continue
                    lsb = mask & -mask
                    derived = table[mask ^ lsb] ^ table[lsb]
                    table[mask] = derived
                    self._derived.append(derived)
                self.nibbles.append(table)

    def retire(self) -> None:
        """Retire all derived parity/accumulator outcome IDs (preserving caller base_outcomes)."""
        for m in self._derived:
            self.b.retire(m)
        self._derived.clear()
        if self._roll is not None:
            self.b.retire(self._roll)
            self._roll = None
            self._roll_words = None
            self._roll_int = 0

    def _handles_for_words(self, words: np.ndarray) -> list:
        if self.use_bytes:
            bt = self.byte_tables
            raw = words.tobytes()
            return [bt[i][bv] for i, bv in enumerate(raw[: self._n_bt]) if bv and bt[i][bv] is not None]
        handles = []
        nibbles = self.nibbles
        n_nib = len(nibbles)
        for w_idx, raw_w in enumerate(words.tolist()):
            val = int(raw_w)
            if not val:
                continue
            base_nib = w_idx << 4
            while val:
                tz_nib = ((val & -val).bit_length() - 1) >> 2
                nib_idx = base_nib + tz_nib
                if nib_idx >= n_nib:
                    break
                shift = tz_nib << 2
                nib_val = (val >> shift) & 0xF
                handles.append(nibbles[nib_idx][nib_val])
                val &= ~(0xF << shift)
        return handles

    def cond_for_words(self, words: np.ndarray):
        if not np.any(words):
            return None
        handles = self._handles_for_words(words)
        if not handles:
            return None
        if len(handles) == 1:
            return handles[0]
        if self._roll is None:
            self._roll = handles[0] ^ handles[1]
            _fast_ixor_outcomes(self.b, self._roll, handles[2:])
            self._roll_words = words.copy()
            return self._roll
        diff = self._roll_words ^ words
        diff_handles = self._handles_for_words(diff)
        _fast_ixor_outcomes(self.b, self._roll, diff_handles)
        self._roll_words = words.copy()
        return self._roll

    def _handles_for_int(self, val: int) -> list:
        handles = []
        if self.use_bytes:
            bt = self.byte_tables
            n_bt = self._n_bt
            rem = val
            while rem:
                tz_byte = ((rem & -rem).bit_length() - 1) >> 3
                if tz_byte >= n_bt:
                    break
                shift = tz_byte << 3
                bv = (rem >> shift) & 0xFF
                h = bt[tz_byte][bv]
                if h is not None:
                    handles.append(h)
                rem &= ~(0xFF << shift)
            return handles
        nibbles = self.nibbles
        n_nib = len(nibbles)
        rem = val
        while rem:
            tz_nib = ((rem & -rem).bit_length() - 1) >> 2
            if tz_nib >= n_nib:
                break
            shift = tz_nib << 2
            nib_val = (rem >> shift) & 0xF
            handles.append(nibbles[tz_nib][nib_val])
            rem &= ~(0xF << shift)
        return handles

    def cond_for_int(self, val: int):
        if not val:
            return None
        if self.use_bytes:
            tz_b = ((val & -val).bit_length() - 1) >> 3
            shifted = val >> (tz_b << 3)
            if shifted <= 0xFF:
                return self.byte_tables[tz_b][shifted] if tz_b < self._n_bt else None
        else:
            tz_n = ((val & -val).bit_length() - 1) >> 2
            shifted = val >> (tz_n << 2)
            if shifted <= 0xF:
                return self.nibbles[tz_n][shifted] if tz_n < len(self.nibbles) else None
        if self._roll is not None:
            diff = self._roll_int ^ val
            if not diff:
                return self._roll
            diff_handles = self._handles_for_int(diff)
            _fast_ixor_outcomes(self.b, self._roll, diff_handles)
            self._roll_int = val
            return self._roll
        handles = self._handles_for_int(val)
        if not handles:
            return None
        if len(handles) == 1:
            return handles[0]
        self._roll = handles[0] ^ handles[1]
        _fast_ixor_outcomes(self.b, self._roll, handles[2:])
        self._roll_int = val
        return self._roll


def emit_const_add(b, target: Sequence[int], constant: int, *, subtract: bool = False, ctrl: int | None = None) -> None:
    n = len(target)
    if n == 0:
        return
    c = (-int(constant) if subtract else int(constant)) % (1 << n)
    if c == 0:
        return
    tz = ctz(c)
    t_slice = list(target[tz:])
    c_shifted = c >> tz
    m = len(t_slice)

    if ctrl is None:
        if m == 1:
            b.x(t_slice[0])
            return
        if m == 2:
            b.cx(t_slice[0], t_slice[1])
            if (c_shifted >> 1) & 1:
                b.x(t_slice[1])
            b.x(t_slice[0])
            return
        trail = []
        carry = t_slice[0]
        for p in range(1, m - 1):
            bit_p = (c_shifted >> p) & 1
            tp = t_slice[p]
            if bit_p == 0:
                anc = b.and_compute(carry, tp)
                trail.append((bit_p, carry, tp, anc))
                carry = anc
            else:
                anc = b.and_compute(carry, tp, pol=(False, False))
                b.x(anc)
                trail.append((bit_p, carry, tp, anc))
                carry = anc
        b.cx(carry, t_slice[m - 1])
        if (c_shifted >> (m - 1)) & 1:
            b.x(t_slice[m - 1])
        for bit_p, prev_c, tp, anc in reversed(trail):
            if bit_p == 0:
                b.and_release(anc, prev_c, tp)
                b.cx(prev_c, tp)
            else:
                b.x(anc)
                b.and_release(anc, prev_c, tp, pol=(False, False))
                b.cx(prev_c, tp)
                b.x(tp)
        b.x(t_slice[0])
        return

    if m == 1:
        b.cx(ctrl, t_slice[0])
        return
    source = [ctrl if ((c_shifted >> p) & 1) else None for p in range(m)]
    _gu_add(b, t_slice, source, shared=ctrl)


def emit_qq_add(b, target: Sequence[int], source: Sequence[int], *, subtract: bool = False) -> None:
    if not target or not source:
        return
    if subtract:
        for q in target:
            b.x(q)
    _gu_add(b, list(target), list(source))
    if subtract:
        for q in target:
            b.x(q)


def _emit_ccx_clean(b, c1: int, c2: int, target: int) -> None:
    t = b.and_compute(c1, c2)
    b.cx(t, target)
    b.and_release(t, c1, c2)


def emit_cuccaro_add(b, target: Sequence[int], source: Sequence[int], *, subtract: bool = False) -> None:
    """Low-space Cuccaro MAJ/UMA ripple ``target +=/-= source (mod 2^len(target))``."""
    m = len(target)
    s = len(source)
    if m == 0 or s == 0:
        return
    if s > m:
        source = list(source[:m])
        s = m
    pad = b.qubits(m - s) if m > s else []
    src = list(source) + list(pad)
    if subtract:
        for q in target:
            b.x(q)
    if m == 1:
        b.cx(src[0], target[0])
    else:
        carry = b.qubit()
        for bit in range(m - 1):
            inc = carry if bit == 0 else src[bit - 1]
            a_q, t_q = src[bit], target[bit]
            b.cx(a_q, t_q)
            b.cx(a_q, inc)
            _emit_ccx_clean(b, t_q, inc, a_q)
        b.cx(src[m - 1], target[m - 1])
        b.cx(src[m - 2], target[m - 1])
        for bit in range(m - 2, -1, -1):
            inc = carry if bit == 0 else src[bit - 1]
            a_q, t_q = src[bit], target[bit]
            _emit_ccx_clean(b, t_q, inc, a_q)
            b.cx(a_q, inc)
            b.cx(inc, t_q)
        b.recycle(carry)
    if subtract:
        for q in target:
            b.x(q)
    for q in reversed(pad):
        b.recycle(q)


def _ripple_single_wire_into(b, high: Sequence[int], delta: int) -> None:
    k = len(high)
    if k == 0:
        return
    if k == 1:
        b.cx(delta, high[0])
        return
    if k == 2:
        b.ccx(delta, high[0], high[1])
        b.cx(delta, high[0])
        return
    trail = []
    c = delta
    for j in range(k - 2):
        anc = b.and_compute(c, high[j])
        trail.append((j, c, anc))
        c = anc
    b.ccx(c, high[k - 2], high[k - 1])
    b.cx(c, high[k - 2])
    for j, prev_c, anc in reversed(trail):
        b.and_release(anc, prev_c, high[j])
        b.cx(prev_c, high[j])


def emit_signed_packed_lane_add(b, target: Sequence[int], word: Sequence[int]) -> None:
    n = len(target)
    m = len(word)
    if m < 1 or m > n:
        raise ValueError(f"Invalid signed word width m={m} for target width n={n}")
    if m == n:
        emit_qq_add(b, target, word)
        return
    if m == 1:
        s = word[0]
        for q in target:
            b.cx(s, q)
        _ripple_single_wire_into(b, target, s)
        for q in target:
            b.cx(s, q)
        return
    low_len = m - 1
    sign = word[m - 1]
    recs = []
    carry = b.and_compute(word[0], target[0])
    recs.append(("g0", 0, word[0], None, carry))
    for p in range(1, low_len):
        wp, tp = word[p], target[p]
        b.cx(carry, wp)
        b.cx(carry, tp)
        anc = b.and_compute(wp, tp)
        b.cx(carry, anc)
        recs.append(("g1", p, wp, carry, anc))
        carry = anc
    high = target[low_len:n]
    for q in high:
        b.cx(sign, q)
    b.cx(sign, carry)
    _ripple_single_wire_into(b, high, carry)
    b.cx(sign, carry)
    for q in high:
        b.cx(sign, q)
    for rec in reversed(recs):
        if rec[0] == "g1":
            _, p, wp, prev_c, anc = rec
            b.cx(prev_c, anc)
            b.and_release(anc, wp, target[p])
            b.cx(prev_c, wp)
            b.cx(wp, target[p])
        else:
            _, p, wp, _, anc = rec
            b.and_release(anc, wp, target[p])
            b.cx(wp, target[p])


def emit_fast_nonrestoring_loop2(
    b, work: Sequence[int], modulus: int, compressed_len: int, *, inverse: bool = False
) -> Sequence[int]:
    """Non-restoring ``ctz(p - 1)``-sliced modular division (``inverse=False``) and uncomputation."""
    n_tot = len(work)
    k = n_tot - compressed_len
    if k <= 0:
        return work[:compressed_len]
    tz = ctz(modulus)
    m_odd = modulus >> tz

    if not inverse:
        j_top = k - 1
        emit_const_add(b, work[j_top + tz : compressed_len + j_top + 1], m_odd, subtract=True)
        for j in range(k - 2, -1, -1):
            s_ctrl = work[compressed_len + j + 1]
            sl = list(work[j + tz : compressed_len + j + 1])
            b.x(s_ctrl)
            for qw in sl:
                b.cx(s_ctrl, qw)
            emit_const_add(b, sl, m_odd, subtract=False)
            for qw in sl:
                b.cx(s_ctrl, qw)
            b.x(s_ctrl)
        s0 = work[compressed_len]
        emit_const_add(b, work[tz:compressed_len], m_odd, subtract=False, ctrl=s0)
    else:
        s0 = work[compressed_len]
        emit_const_add(b, work[tz:compressed_len], m_odd, subtract=True, ctrl=s0)
        for j in range(0, k - 1):
            s_ctrl = work[compressed_len + j + 1]
            sl = list(work[j + tz : compressed_len + j + 1])
            b.x(s_ctrl)
            for qw in sl:
                b.cx(s_ctrl, qw)
            emit_const_add(b, sl, m_odd, subtract=True)
            for qw in sl:
                b.cx(s_ctrl, qw)
            b.x(s_ctrl)
        j_top = k - 1
        emit_const_add(b, work[j_top + tz : compressed_len + j_top + 1], m_odd, subtract=False)
    return work[:compressed_len]


def materialize_anf_subset_tree(b, wires: Sequence[int]) -> tuple[dict[int, int | None], list[tuple[int, int, int]]]:
    """Materialize all subset monomials ``0 .. (1<<n)-1`` on ``wires``."""
    n = len(wires)
    m2w: dict[int, int | None] = {0: None}
    for i in range(n):
        m2w[1 << i] = wires[i]
    trail: list[tuple[int, int, int]] = []
    for mask in range(3, 1 << n):
        if (mask & (mask - 1)) == 0:
            continue
        hi = 1 << (mask.bit_length() - 1)
        wa = m2w[mask ^ hi]
        wb = m2w[hi]
        assert wa is not None and wb is not None
        wo = b.and_compute(wa, wb)
        m2w[mask] = wo
        trail.append((wo, wa, wb))
    return m2w, trail


def release_anf_subset_tree(b, trail: Sequence[tuple[int, int, int]]) -> None:
    for wo, wa, wb in reversed(trail):
        b.and_release(wo, wa, wb)


def emit_anf_qrom(b, address: Sequence[int], table: Sequence[int], target: Sequence[int], *, is_anf: bool = False) -> None:
    n = len(address)
    if len(table) != (1 << n):
        raise ValueError("ANF QROM table length mismatch")
    anf = list(table) if is_anf else mobius_transform_int_list(table, n)
    if n == 0:
        xor_word(b, None, anf[0], target)
        return
    has_desc = [False] * (1 << n)
    for mask in range((1 << n) - 1, 0, -1):
        msb = mask.bit_length() - 1
        has_desc[mask] = any(anf[mask | (1 << nxt)] or has_desc[mask | (1 << nxt)] for nxt in range(msb + 1, n))
    if anf[0]:
        xor_word(b, None, anf[0], target)

    def dfs(mask: int, wire: int, min_next: int) -> None:
        if anf[mask]:
            xor_word(b, wire, anf[mask], target)
        if not has_desc[mask]:
            return
        for nxt in range(min_next, n):
            cm = mask | (1 << nxt)
            if not anf[cm] and not has_desc[cm]:
                continue
            cw = b.and_compute(wire, address[nxt])
            dfs(cm, cw, nxt + 1)
            b.and_release(cw, wire, address[nxt])

    for i in range(n):
        m1 = 1 << i
        if anf[m1] or has_desc[m1]:
            dfs(m1, address[i], i + 1)


def emit_bilinear_qrom_load(
    b,
    low_wires: Sequence[int],
    high_map: dict[int, int | None],
    n_high: int,
    table: Sequence[int],
    target: Sequence[int],
    *,
    keep_low_map: bool = False,
) -> tuple[list[int], dict[int, int | None], list[tuple[int, int, int]]]:
    """Load ``target ^= table[low_wires + high_wires]`` reusing pre-materialized ``high_map``.

    Uses at most ``(2^n_low - n_low - 1) + 1`` extra ancilla qubits above ``high_map``!
    If ``keep_low_map=True``, returns ``(anf, low_map, low_trail)`` with ``low_map`` still live
    for immediate 0-Toffoli ``CZ(low_map, high_map, cond=...)`` phase cleanup!
    """
    n_low = len(low_wires)
    n_tot = n_low + n_high
    anf = mobius_transform_int_list(table, n_tot)
    low_map, low_trail = materialize_anf_subset_tree(b, low_wires)

    for m_hi in range(1 << n_high):
        w_hi = high_map[m_hi]
        base = m_hi << n_low
        for m_lo in range(1 << n_low):
            word = anf[base | m_lo]
            if not word:
                continue
            w_lo = low_map[m_lo]
            if w_lo is None and w_hi is None:
                xor_word(b, None, word, target)
            elif w_hi is None:
                xor_word(b, w_lo, word, target)
            elif w_lo is None:
                xor_word(b, w_hi, word, target)
            else:
                cw = b.and_compute(w_lo, w_hi)
                xor_word(b, cw, word, target)
                b.and_release(cw, w_lo, w_hi)

    if not keep_low_map:
        release_anf_subset_tree(b, low_trail)
        return anf, {}, []
    return anf, low_map, low_trail


def emit_bilinear_zero_phase_cleanup(
    b,
    anf_words: np.ndarray,
    n_low: int,
    n_high: int,
    low_map: dict[int, int | None],
    high_map: dict[int, int | None],
    outcomes: Sequence,
) -> None:
    """Apply exact X-measurement phase cleanup on ``low_map x high_map`` using ZERO Toffolis."""
    if not outcomes or not np.any(anf_words):
        for m in outcomes:
            b.retire(m)
        return
    synth = NibbleOutcomeSynthesizer(b, outcomes)
    for m_hi in range(1 << n_high):
        w_hi = high_map[m_hi]
        base = m_hi << n_low
        for m_lo in range(1 << n_low):
            cond = synth.cond_for_words(anf_words[base | m_lo])
            if cond is None:
                continue
            w_lo = low_map[m_lo]
            if w_lo is None and w_hi is None:
                b.neg(cond=cond)
            elif w_hi is None:
                b.z(w_lo, cond=cond)
            elif w_lo is None:
                b.z(w_hi, cond=cond)
            else:
                b.cz(w_lo, w_hi, cond=cond)
    synth.retire()
    for m in outcomes:
        b.retire(m)


def emit_selectswap_shared_sel(
    b,
    src_wires: Sequence[int],
    sel_wires: Sequence[int],
    sel_map: dict[int, int | None] | None,
    table: Sequence[int],
    width: int,
    lam: int,
):
    """SelectSwap QROAM on ``address = src_wires + sel_wires`` reusing pre-materialized ``sel_map``
    whenever ``sel_wires`` sit cleanly inside ``hbits = address[log_lam:]``.
    """
    address = list(src_wires) + list(sel_wires)
    n = len(address)
    if len(table) != (1 << n):
        raise ValueError("Table size mismatch")
    lam = min(lam, 1 << n)
    log_lam = lam.bit_length() - 1
    lbits = address[:log_lam]
    hbits = address[log_lam:]
    nh = len(hbits)
    n_chunks = 1 << nh

    banks = [b.qubits(width) for _ in range(lam)]
    packed_chunks = []
    mask_w = (1 << width) - 1
    for hv in range(n_chunks):
        base = hv << log_lam
        val = 0
        for j in range(lam):
            val |= (int(table[base | j]) & mask_w) << (j * width)
        packed_chunks.append(val)
    flat_target = [q for bank in banks for q in bank]

    n_sel = len(sel_wires)
    if sel_map is not None and log_lam <= len(src_wires) and nh <= 10:
        low_h = list(src_wires[log_lam:])
        emit_bilinear_qrom_load(b, low_h, sel_map, n_sel, packed_chunks, flat_target, keep_low_map=False)
    elif nh <= 9:
        emit_anf_qrom(b, hbits, packed_chunks, flat_target)
    else:
        def select_leaf(ctrl, hv):
            base = hv << log_lam
            for j in range(lam):
                xor_word(b, ctrl, int(table[base | j]) & mask_w, banks[j])
        walk(b, hbits, None, nh - 1, 0, n_chunks, 0, n_chunks, select_leaf)

    for i in range(log_lam):
        step = 1 << i
        span = step << 1
        for s in range(0, lam, span):
            hi_slot = s + step
            for t in range(width):
                b.cswap(lbits[i], banks[s][t], banks[hi_slot][t])

    perm = swap_perm(lam)

    def measure_banks(*, include_bank0: bool):
        start_j = 0 if include_bank0 else 1
        measured_banks = banks[start_j:]
        if not measured_banks:
            return [], [0] * (1 << n)
        bank_outcomes = [[b.hmr(q) for q in bank] for bank in measured_banks]
        for bank in reversed(measured_banks):
            for q in reversed(bank):
                b.recycle(q)
        flat_outcomes = [m for bo in bank_outcomes for m in bo]
        addr_masks = [0] * (1 << n)
        for av in range(1 << n):
            hv, lv = av >> log_lam, av & (lam - 1)
            base = hv << log_lam
            prow = perm[lv]
            m_int = 0
            for idx_b, j in enumerate(range(start_j, lam)):
                m_int |= (int(table[base | prow[j]]) & mask_w) << (idx_b * width)
            addr_masks[av] = m_int
        return flat_outcomes, addr_masks

    return banks[0], measure_banks


def _build_disjoint_parity_prep(i1: Sequence[int], i2: Sequence[int]) -> tuple[int, int, list[tuple[int, int]]]:
    s1, s2 = set(i1), set(i2)
    only1, only2 = sorted(s1 - s2), sorted(s2 - s1)
    cnots: list[tuple[int, int]] = []
    if only1 and only2:
        wa, wb = only1[0], only2[0]
        for a in sorted(s1):
            if a != wa:
                cnots.append((a, wa))
        for c in sorted(s2):
            if c != wb:
                cnots.append((c, wb))
        return wa, wb, cnots
    if not only1:
        wa, wb = sorted(s1)[0], only2[0]
        for a in sorted(s1):
            if a != wa:
                cnots.append((a, wa))
        for c in only2:
            if c != wb:
                cnots.append((c, wb))
        cnots.append((wa, wb))
        return wa, wb, cnots
    wb, wa = sorted(s2)[0], only1[0]
    for c in sorted(s2):
        if c != wb:
            cnots.append((c, wb))
    for a in only1:
        if a != wa:
            cnots.append((a, wa))
    cnots.append((wb, wa))
    return wa, wb, cnots


class _FrozenUniversalCZPlan:
    def __init__(self, n: int, recipes: list[tuple[list[int], list[int]]]):
        self.n = n
        self.N = 1 << n
        self.recipes = recipes
        self.K = len(recipes)
        self.target_idx = [m for m in range(self.N) if m.bit_count() >= 3]
        self.D = len(self.target_idx)

        wires = [((np.arange(self.N) >> i) & 1).astype(np.uint8) for i in range(n)]
        for i1, i2 in recipes:
            u = np.bitwise_xor.reduce([wires[idx] for idx in i1])
            v = np.bitwise_xor.reduce([wires[idx] for idx in i2])
            wires.append(u & v)
        self.wire_tts = wires

        L = len(wires)
        cz_pairs: list[tuple[int, int]] = []
        rows: list[np.ndarray] = []
        for a in range(n, L):
            for c in range(a):
                cz_pairs.append((a, c))
                prod = (wires[a] & wires[c]).astype(np.uint64).reshape(self.N, 1)
                anf_p = mobius_transform_u64_matrix(prod, n)[:, 0].astype(np.uint8)
                rows.append(anf_p[self.target_idx])
        M = np.array(rows, dtype=np.uint8)
        num_cz = len(cz_pairs)

        self.echelon_pivs: list[int] = []
        self.echelon_vecs: dict[int, np.ndarray] = {}
        self.echelon_combs: dict[int, np.ndarray] = {}
        for r in range(num_cz):
            v = M[r].copy()
            comb = np.zeros(num_cz, dtype=np.uint8)
            comb[r] = 1
            for p in self.echelon_pivs:
                if v[p]:
                    v ^= self.echelon_vecs[p]
                    comb ^= self.echelon_combs[p]
            nz = np.flatnonzero(v)
            if len(nz):
                pn = int(nz[0])
                self.echelon_vecs[pn] = v
                self.echelon_combs[pn] = comb
                self.echelon_pivs.append(pn)
                self.echelon_pivs.sort()
                if len(self.echelon_pivs) == self.D:
                    break
        if len(self.echelon_pivs) != self.D:
            raise RuntimeError(f"Frozen CZ plan rank {len(self.echelon_pivs)} != {self.D} for n={n}")
        self.cz_pairs = cz_pairs


_FROZEN_CZ_PLANS: dict[int, _FrozenUniversalCZPlan] = {}


def _get_frozen_cz_plan(n: int) -> _FrozenUniversalCZPlan:
    plan = _FROZEN_CZ_PLANS.get(n)
    if plan is None:
        plan = _FrozenUniversalCZPlan(n, FROZEN_UNIVERSAL_CZ_RECIPES[n])
        _FROZEN_CZ_PLANS[n] = plan
    return plan


def emit_fast_z_mask_cleanup(
    b,
    m2w: dict[int, int | None],
    anf_words: np.ndarray,
    base_outcomes: Sequence,
    cnt: int,
) -> None:
    if anf_words.shape[1] == 0 or not np.any(anf_words):
        return
    synth = NibbleOutcomeSynthesizer(b, base_outcomes, use_bytes=True)
    for mask in range(cnt):
        cond = synth.cond_for_words(anf_words[mask])
        if cond is not None:
            if mask == 0:
                b.neg(cond=cond)
            else:
                b.z(m2w[mask], cond=cond)
    synth.retire()


def emit_asymmetric_half_dfs_phaseup(
    b,
    address: Sequence[int],
    anf_outcome_words: np.ndarray,
    base_outcomes: Sequence,
    *,
    n1_override: int | None = None,
) -> None:
    """Half-materialized / half-DFS-streamed phase synthesis with optional asymmetric ``n1_override``."""
    n = len(address)
    if anf_outcome_words.shape[0] != (1 << n):
        raise ValueError("ANF outcome matrix row count mismatch")
    if anf_outcome_words.shape[1] == 0 or not np.any(anf_outcome_words):
        return
    synth = NibbleOutcomeSynthesizer(b, base_outcomes, use_bytes=(len(base_outcomes) >= 8))
    cols = [anf_outcome_words[:, w_i].tolist() for w_i in range(anf_outcome_words.shape[1])]
    row_ints = cols[0]
    for w_i in range(1, len(cols)):
        sh = w_i * 64
        row_ints = [a | (b_v << sh) for a, b_v in zip(row_ints, cols[w_i])]

    def apply_on_pair(w2: int | None, w1: int | None, idx: int) -> None:
        cond = synth.cond_for_int(row_ints[idx])
        if cond is None:
            return
        if w2 is None and w1 is None:
            b.neg(cond=cond)
        elif w2 is None:
            b.z(w1, cond=cond)
        elif w1 is None:
            b.z(w2, cond=cond)
        else:
            b.cz(w2, w1, cond=cond)

    if n <= 2:
        apply_on_pair(None, None, 0)
        if n >= 1:
            apply_on_pair(None, address[0], 1)
        if n == 2:
            apply_on_pair(address[1], None, 2)
            apply_on_pair(address[1], address[0], 3)
        synth.retire()
        return

    n1 = n1_override if (n1_override is not None and 1 <= n1_override < n) else (n >> 1)
    n2 = n - n1
    h1 = list(address[:n1])
    h2 = list(address[n1:])
    m1_size = 1 << n1
    m2_size = 1 << n2

    nz = np.any(anf_outcome_words != 0, axis=1).reshape(m2_size, m1_size)
    row_nonzero = np.any(nz, axis=1)
    col_nonzero = np.any(nz, axis=0)

    p1_needed = [False] * m1_size
    for mask in range(m1_size - 1, 0, -1):
        msb = mask.bit_length() - 1
        p1_needed[mask] = bool(col_nonzero[mask]) or any(p1_needed[mask | (1 << nxt)] for nxt in range(msb + 1, n1))

    p1_wires: dict[int, int | None] = {0: None}
    p1_trail: list[tuple[int, int, int]] = []
    for i in range(n1):
        p1_wires[1 << i] = h1[i]
    for mask in range(3, m1_size):
        if (mask & (mask - 1)) == 0 or not p1_needed[mask]:
            continue
        msb = mask.bit_length() - 1
        prev = mask ^ (1 << msb)
        w = b.and_compute(p1_wires[prev], h1[msb])
        p1_wires[mask] = w
        p1_trail.append((w, p1_wires[prev], h1[msb]))

    active_m1 = [m1 for m1 in range(m1_size) if col_nonzero[m1]]

    p2_has_desc = [False] * m2_size
    for mask in range(m2_size - 1, 0, -1):
        msb = mask.bit_length() - 1
        p2_has_desc[mask] = any(row_nonzero[mask | (1 << nxt)] or p2_has_desc[mask | (1 << nxt)] for nxt in range(msb + 1, n2))

    def emit_row(m2: int, w2: int | None) -> None:
        if not row_nonzero[m2]:
            return
        base = m2 << n1
        for m1 in active_m1:
            apply_on_pair(w2, p1_wires[m1], base | m1)

    emit_row(0, None)

    def dfs_p2(mask: int, wire: int, min_next: int) -> None:
        emit_row(mask, wire)
        if not p2_has_desc[mask]:
            return
        for nxt in range(min_next, n2):
            cm = mask | (1 << nxt)
            if not row_nonzero[cm] and not p2_has_desc[cm]:
                continue
            cw = b.and_compute(wire, h2[nxt])
            dfs_p2(cm, cw, nxt + 1)
            b.and_release(cw, wire, h2[nxt])

    for j in range(n2):
        m2 = 1 << j
        if row_nonzero[m2] or p2_has_desc[m2]:
            dfs_p2(m2, h2[j], j + 1)

    for w, c1, c2 in reversed(p1_trail):
        b.and_release(w, c1, c2)
    synth.retire()


def emit_universal_cz_phaseup(
    b,
    address: Sequence[int],
    tt_outcome_words: np.ndarray,
    base_outcomes: Sequence,
    *,
    n1_override: int | None = None,
) -> None:
    """Static outcome-independent phase synthesis:
    - ``n <= 4``: optimal ``0, 1, 2`` Toffoli Half-DFS
    - ``n in {5, 6, 7, 8}``: frozen Universal All-to-All CZ DAG (``4, 7, 11, 17`` Toffolis!)
    - ``n >= 9``: ``emit_asymmetric_half_dfs_phaseup`` (with ``n1_override`` for Q-safe ``k=1``)
    """
    n = len(address)
    if not base_outcomes or tt_outcome_words.shape[1] == 0 or not np.any(tt_outcome_words):
        return
    anf_w = mobius_transform_u64_matrix(tt_outcome_words, n)
    if n <= 4 or n >= 9:
        emit_asymmetric_half_dfs_phaseup(b, address, anf_w, base_outcomes, n1_override=n1_override)
        return

    plan = _get_frozen_cz_plan(n)
    rem = anf_w[plan.target_idx].copy()
    if not np.any(rem):
        # Pure degree <= 2 Clifford table: 0 Toffolis!
        emit_asymmetric_half_dfs_phaseup(b, address[:2], anf_w[:4], base_outcomes)
        nsyn = NibbleOutcomeSynthesizer(b, base_outcomes)
        for i in range(n):
            if i >= 2:
                ci = nsyn.cond_for_words(anf_w[1 << i])
                if ci is not None:
                    b.z(address[i], cond=ci)
            for j in range(max(i + 1, 2), n):
                cij = nsyn.cond_for_words(anf_w[(1 << i) | (1 << j)])
                if cij is not None:
                    b.cz(address[i], address[j], cond=cij)
        nsyn.retire()
        return

    comb_words = np.zeros((len(plan.cz_pairs), rem.shape[1]), dtype=np.uint64)
    for p in plan.echelon_pivs:
        row_w = rem[p].copy()
        if np.any(row_w):
            v_mask = plan.echelon_vecs[p].astype(np.uint64)[:, None]
            c_mask = plan.echelon_combs[p].astype(np.uint64)[:, None]
            rem ^= v_mask * row_w[None, :]
            comb_words ^= c_mask * row_w[None, :]

    tt_rem = tt_outcome_words.copy()
    for cz_i in range(len(plan.cz_pairs)):
        cw = comb_words[cz_i]
        if np.any(cw):
            wa_i, wb_i = plan.cz_pairs[cz_i]
            prod_mask = (plan.wire_tts[wa_i] & plan.wire_tts[wb_i]).astype(np.uint64)[:, None]
            tt_rem ^= prod_mask * cw[None, :]
    anf_rem = mobius_transform_u64_matrix(tt_rem, n)

    nsyn = NibbleOutcomeSynthesizer(b, base_outcomes)
    wmap: dict[int, int] = {i: address[i] for i in range(n)}
    uncomp: list[tuple[int, int, int, list[tuple[int, int]]]] = []
    for r_idx, (i1, i2) in enumerate(plan.recipes):
        wa, wb, prep_c = _build_disjoint_parity_prep(i1, i2)
        for src_i, dst_i in prep_c:
            b.cx(wmap[src_i], wmap[dst_i])
        w_out = b.and_compute(wmap[wa], wmap[wb])
        for src_i, dst_i in reversed(prep_c):
            b.cx(wmap[src_i], wmap[dst_i])
        wmap[n + r_idx] = w_out
        uncomp.append((w_out, wa, wb, prep_c))

    for cz_i in range(len(plan.cz_pairs)):
        cond = nsyn.cond_for_words(comb_words[cz_i])
        if cond is not None:
            wa_i, wb_i = plan.cz_pairs[cz_i]
            b.cz(wmap[wa_i], wmap[wb_i], cond=cond)

    for w_out, wa, wb, prep_c in reversed(uncomp):
        for src_i, dst_i in prep_c:
            b.cx(wmap[src_i], wmap[dst_i])
        b.and_release(w_out, wmap[wa], wmap[wb])
        for src_i, dst_i in reversed(prep_c):
            b.cx(wmap[src_i], wmap[dst_i])

    c0 = nsyn.cond_for_words(anf_rem[0])
    if c0 is not None:
        b.neg(cond=c0)
    for i in range(n):
        ci = nsyn.cond_for_words(anf_rem[1 << i])
        if ci is not None:
            b.z(address[i], cond=ci)
        for j in range(i + 1, n):
            cij = nsyn.cond_for_words(anf_rem[(1 << i) | (1 << j)])
            if cij is not None:
                b.cz(address[i], address[j], cond=cij)
    nsyn.retire()
