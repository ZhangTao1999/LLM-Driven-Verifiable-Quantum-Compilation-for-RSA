"""RSA-2048 Batched RNS + Dyadic Torus Flat OPSTRM01 Compiler.

Integrates non-overlapping verified optimizations into clean-boundary flat ``OPSTRM01``
gate streams certified by BOTH symbolic all-2^M measurement-outcome phase tracking AND
randomized ``kickmix`` replay:

1. Power-of-2 Dyadic Fixed-Point CRT (``mod 2^f``, eliminating the borrow qubit,
   cutting the worst-case coset-mask CRT error diameter by one-third at ``g_gap = f``,
   and eliminating dual-lookup + borrow comparator via bilinear ``low_map x high_map``
   zero-phase ANF accumulation).
2. Non-Restoring ``ctz(p - 1)``-Sliced ``loop2`` / ``unloop2`` (1 uncontrolled conjugated
   constant adder per middle division bit instead of 2 restoring adders).
3. Shared-Selector ``loop3`` (reusing ``selector`` ANF wires across all ``7-8`` source
   chunks inside both ``SelectSwap(lam=2)`` and ``ANF`` QROM) + Bilinear Zero-Phase
   Reverse ``unloop3``.
4. Zero-Phase, Zero-Boundary-Tax ``RollingANF_Loop1`` (for ``w1=8, k>=12`` and bilinear
   ``3b x 3b`` for ``w1=6, k=1``; automatic ``ZeroTail_Packed_Loop1`` for ``w1=9``
   where a ``21*k`` packed buffer stays narrower than the 9-bit rolling workspace).
5. Static Symbolic-Outcome ``UniversalAllToAllCZ`` (``n=5..8`` -> ``4, 7, 11, 17`` Toffolis,
   used on packed-mode tail windows and standalone phase tables) and configurable ``(n1, n2)``
   Half-DFS (including Q-safe ``(6, 10)`` at ``a=16`` for ``k=1`` -> ``1,306 Q``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from flat_stream import Builder

from compiler.primitives import (
    compute_all_crt_pairs,
    dyadic_crt_entry_value,
    emit_anf_qrom,
    emit_bilinear_qrom_load,
    emit_bilinear_zero_phase_cleanup,
    emit_const_add,
    emit_cuccaro_add,
    emit_fast_nonrestoring_loop2,
    emit_fast_z_mask_cleanup,
    emit_qq_add,
    emit_selectswap_shared_sel,
    emit_signed_packed_lane_add,
    emit_universal_cz_phaseup,
    fast_fanout_anf_lane,
    materialize_anf_subset_tree,
    mobius_transform_int_list,
    mobius_transform_u64_matrix,
    primitive_root_prime,
    release_anf_subset_tree,
    words_from_int_masks,
)


@dataclass(frozen=True)
class PublicRNSProblem:
    name: str
    modulus: int
    generator: int
    num_input_qubits: int
    window1: int
    len_accumulator: int
    mask_bits: int
    min_wraparound_gap: int
    prime_bits: int
    periods: tuple[int, ...]
    generators: tuple[int, ...]
    crt_pairs: tuple[tuple[int, int], ...]  # ((L/p_i)^-1 mod p_i, (L/p_i) mod N)
    full_rns_product: int
    input_sha256: str

    @property
    def num_windows1(self) -> int:
        return (self.num_input_qubits + self.window1 - 1) // self.window1


@dataclass(frozen=True)
class CombinedScheduleSpec:
    variant_id: str
    s_param: int
    share: int
    center_bits: int
    len_accumulator: int = 33
    mask_bits: int = 26
    min_wraparound_gap: int = 33
    representation: str = "shifted_unsigned"
    b0: int = 10
    stage_widths: tuple[int, ...] = (3, 3, 2)
    selector_window: int = 3
    source_window: int = 3
    terminal_crt: bool = True
    terminal_sel_bits: int = 2
    cutoff: int = 2
    partial_width: bool = True
    startup_lam: int = 8
    stage_lam: int = 2
    loop1_mode: str = "rolling_anf"  # "rolling_anf", "bilinear_rolling_anf", or "packed_zero_tail"
    q_safe_k1: bool = False
    phase_n1_override: int | None = None


R4B_PLUS_COMBINED_SPEC = CombinedScheduleSpec(
    variant_id="R4B_PLUS_COMBINED",
    s_param=4,
    share=13,
    center_bits=26,
    len_accumulator=33,
    mask_bits=26,
    min_wraparound_gap=33,
    b0=9,
    stage_widths=(3, 3, 2),
    terminal_sel_bits=3,
    cutoff=2,
    startup_lam=4,
    stage_lam=2,
    loop1_mode="rolling_anf",
    q_safe_k1=False,
)

S4_ROLLK20_COMBINED_SPEC = CombinedScheduleSpec(
    variant_id="S4_ROLLK20_COMBINED",
    s_param=4,
    share=20,
    center_bits=26,
    len_accumulator=33,
    mask_bits=26,
    min_wraparound_gap=33,
    b0=9,
    stage_widths=(3, 3, 2),
    terminal_sel_bits=3,
    cutoff=2,
    startup_lam=4,
    stage_lam=2,
    loop1_mode="rolling_anf",
    q_safe_k1=False,
)

S3_K20_W9_COMBINED_SPEC = CombinedScheduleSpec(
    variant_id="S3_K20_W9_COMBINED",
    s_param=3,
    share=20,
    center_bits=27,
    len_accumulator=34,
    mask_bits=27,
    min_wraparound_gap=34,
    b0=9,
    stage_widths=(3, 3, 2),
    terminal_sel_bits=3,
    cutoff=2,
    startup_lam=4,
    stage_lam=2,
    loop1_mode="packed_zero_tail",
    q_safe_k1=False,
)

S14_PREFIX_COMBINED_SPEC = CombinedScheduleSpec(
    variant_id="S14_PREFIX_COMBINED",
    s_param=14,
    share=1,
    center_bits=26,
    len_accumulator=33,
    mask_bits=26,
    min_wraparound_gap=33,
    b0=8,
    stage_widths=(3, 3, 2),
    terminal_sel_bits=4,
    cutoff=3,
    startup_lam=2,
    stage_lam=1,
    loop1_mode="bilinear_rolling_anf",
    q_safe_k1=True,
    phase_n1_override=6,
)

S3_K22_W9_COMBINED_SPEC = CombinedScheduleSpec(
    variant_id="S3_K22_W9_COMBINED",
    s_param=3,
    share=22,
    center_bits=27,
    len_accumulator=34,
    mask_bits=27,
    min_wraparound_gap=34,
    b0=9,
    stage_widths=(3, 3, 2),
    terminal_sel_bits=3,
    cutoff=2,
    startup_lam=4,
    stage_lam=2,
    loop1_mode="packed_zero_tail",
    q_safe_k1=False,
)

S4_K22_W8_CMB_SPEC = CombinedScheduleSpec(
    variant_id="S4_K22_W8_CMB",
    s_param=4,
    share=22,
    center_bits=26,
    len_accumulator=33,
    mask_bits=26,
    min_wraparound_gap=33,
    b0=10,
    stage_widths=(3, 3, 2),
    terminal_sel_bits=2,
    cutoff=2,
    startup_lam=8,
    stage_lam=2,
    loop1_mode="rolling_anf",
    q_safe_k1=False,
)

S4_K24_W8_OPT_SPEC = CombinedScheduleSpec(
    variant_id="S4_K24_W8_OPT",
    s_param=4,
    share=24,
    center_bits=26,
    len_accumulator=33,
    mask_bits=26,
    min_wraparound_gap=33,
    b0=9,
    stage_widths=(3, 3, 2),
    terminal_sel_bits=3,
    cutoff=2,
    startup_lam=4,
    stage_lam=2,
    loop1_mode="rolling_anf",
    q_safe_k1=False,
)

COMBINED_SPECS_BY_NAME: dict[str, CombinedScheduleSpec] = {
    s.variant_id: s
    for s in (
        R4B_PLUS_COMBINED_SPEC,
        S4_ROLLK20_COMBINED_SPEC,
        S4_K22_W8_CMB_SPEC,
        S4_K24_W8_OPT_SPEC,
        S3_K20_W9_COMBINED_SPEC,
        S3_K22_W9_COMBINED_SPEC,
        S14_PREFIX_COMBINED_SPEC,
    )
}


def _parse_ini_fields(raw: str | dict) -> dict:
    if isinstance(raw, dict):
        return raw
    import ast
    out = {}
    for line in raw.splitlines():
        line = line.split(";", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            try:
                out[k.strip()] = ast.literal_eval(v.strip())
            except (ValueError, SyntaxError):
                out[k.strip()] = v.strip()
    return out


@lru_cache(maxsize=8)
def _load_public_rns_problem_cached(
    resolved_path_str: str,
    max_primes: int | None,
    name: str,
    override_f: int | None,
    override_mask: int | None,
    override_gap: int | None,
) -> PublicRNSProblem:
    path = Path(resolved_path_str)
    raw_bytes = path.read_bytes()
    h_sha = hashlib.sha256(raw_bytes).hexdigest()
    data = json.loads(raw_bytes.decode("utf-8"))
    fields = _parse_ini_fields(data.get("problem_fields", data.get("problem", data.get("problem_config", data))))
    modulus = int(fields.get("modulus", data.get("modulus")))
    generator = int(fields.get("generator", 3))
    num_input_qubits = int(fields["num_input_qubits"])
    window1 = int(fields["window1"])
    len_acc = int(override_f if override_f is not None else fields.get("len_accumulator", fields.get("accumulator_bits", 33)))
    mask_bits = int(override_mask if override_mask is not None else fields.get("mask_bits", 26))
    gap = int(override_gap if override_gap is not None else fields.get("min_wraparound_gap", fields.get("wraparound_gap", 33)))
    prime_bits = int(fields.get("rns_primes_bit_length", fields.get("prime_bits", 20)))

    all_periods = tuple(map(int, data.get("full_rns_periods", data.get("periods"))))
    full_product = math.prod(all_periods)
    all_crt_pairs = compute_all_crt_pairs(all_periods, modulus)

    periods = all_periods[:max_primes] if max_primes is not None else all_periods
    crt_pairs = all_crt_pairs[:max_primes] if max_primes is not None else all_crt_pairs
    try:
        from sympy import primitive_root as _sym_prim_root  # type: ignore
        gens = tuple(int(_sym_prim_root(p)) for p in periods)
    except ImportError:
        gens = tuple(primitive_root_prime(p) for p in periods)

    return PublicRNSProblem(
        name=name,
        modulus=modulus,
        generator=generator,
        num_input_qubits=num_input_qubits,
        window1=window1,
        len_accumulator=len_acc,
        mask_bits=mask_bits,
        min_wraparound_gap=gap,
        prime_bits=prime_bits,
        periods=periods,
        generators=gens,
        crt_pairs=crt_pairs,
        full_rns_product=full_product,
        input_sha256=h_sha,
    )


def load_public_rns_problem(
    input_json_path: str | Path,
    *,
    max_primes: int | None = None,
    name: str = "S4_P20_W8",
    spec: CombinedScheduleSpec | None = None,
) -> PublicRNSProblem:
    """Load public mathematical RNS JSON strictly read-only."""
    resolved = str(Path(input_json_path).resolve())
    # Only override f/mask/gap on full RSA-2048 moduli (validation330 keeps its native 330-bit fields).
    raw_peek = json.loads(Path(resolved).read_text(encoding="utf-8"))
    f_peek = _parse_ini_fields(raw_peek.get("problem_fields", raw_peek.get("problem", raw_peek.get("problem_config", raw_peek))))
    is_2048 = int(f_peek.get("modulus", raw_peek.get("modulus", 0))).bit_length() >= 2000
    ov_f = spec.len_accumulator if (spec is not None and is_2048) else None
    ov_m = spec.mask_bits if (spec is not None and is_2048) else None
    ov_g = spec.min_wraparound_gap if (spec is not None and is_2048) else None
    return _load_public_rns_problem_cached(resolved, max_primes, name, ov_f, ov_m, ov_g)


@lru_cache(maxsize=8)
def multiplier_table(modulus: int, generator: int, num_windows1: int, window1: int) -> tuple[tuple[int, ...], ...]:
    rows = []
    base = generator % modulus
    for _ in range(num_windows1):
        vals = [1]
        for _ in range(1, 1 << window1):
            vals.append((vals[-1] * base) % modulus)
        rows.append(tuple(vals))
        base = pow(base, 1 << window1, modulus)
    return tuple(rows)


def compute_prime_dlog_row(prob: PublicRNSProblem, prime_idx: int) -> np.ndarray:
    p = prob.periods[prime_idx]
    g = prob.generators[prime_idx]
    logs = np.empty(p, dtype=np.uint32)
    logs[0] = 0xFFFFFFFF
    val = 1
    for exp in range(p - 1):
        logs[val] = exp
        val = (val * g) % p
    mults = multiplier_table(prob.modulus, prob.generator, prob.num_windows1, prob.window1)
    out = np.empty((prob.num_windows1, 1 << prob.window1), dtype=np.uint32)
    for w_idx, row in enumerate(mults):
        rem = [v % p for v in row]
        out[w_idx, :] = logs[rem]
    return out


def log_center_for_row(row: np.ndarray, num_input_qubits: int, window1: int) -> int:
    arr = np.asarray(row, dtype=np.uint64)
    denom = 1 << window1
    full_w = min(arr.shape[0], num_input_qubits // window1)
    num = int(arr[:full_w, :denom].sum()) if full_w else 0
    for w_idx in range(full_w, arr.shape[0]):
        width = min(window1, num_input_qubits - w_idx * window1)
        cnt = 1 << width
        num += int(arr[w_idx, :cnt].sum()) * (denom // cnt)
    return (num + denom // 2) // denom


def startup_offset_for_row(row: np.ndarray, p: int, prob: PublicRNSProblem, spec: CombinedScheduleSpec) -> int:
    center = log_center_for_row(row, prob.num_input_qubits, prob.window1)
    if spec.representation == "shifted_unsigned":
        exp = spec.center_bits - p.bit_length() - 1
        if exp < 0:
            raise ValueError("center_bits too small for shifted_unsigned")
        half = (1 << exp) * (p - 1)
        return center - half
    return center


def resolve_prime_stages(dlog_bits: int, spec: CombinedScheduleSpec) -> tuple[int, list[tuple[int, int]], tuple[int, int] | None]:
    b0_bits = min(spec.b0, dlog_bits)
    rem = dlog_bits - b0_bits
    if rem == 0:
        return b0_bits, [], None
    term_w = min(spec.terminal_sel_bits, rem) if spec.terminal_crt else 0
    mid_rem = rem - term_w
    stages: list[tuple[int, int]] = []
    cur = b0_bits
    for w in spec.stage_widths:
        if mid_rem <= 0:
            break
        take = min(w, mid_rem)
        stages.append((cur, take))
        cur += take
        mid_rem -= take
    while mid_rem > 0:
        take = min(spec.selector_window, mid_rem)
        stages.append((cur, take))
        cur += take
        mid_rem -= take
    term_stage = (cur, term_w) if term_w > 0 else None
    return b0_bits, stages, term_stage


def residue_width_for(prime_bits: int, source_window: int) -> int:
    w = prime_bits
    while (w + source_window - 1) // source_window > (1 << (w - prime_bits)):
        w += 1
    return w


def positive_table(p: int, g: int, sel_offset: int, src_offset: int, sel_bits: int, src_bits: int) -> list[int]:
    out = []
    src_scale = pow(2, src_offset, p)
    for sel in range(1 << sel_bits):
        mult = (pow(g, sel << sel_offset, p) * src_scale) % p
        out.extend((mult * val) % p for val in range(1 << src_bits))
    return out


def oracle_prime_dyadic_delta(
    prob: PublicRNSProblem,
    prime_idx: int,
    dlog_sum: int,
    offset_b: int,
    spec: CombinedScheduleSpec,
) -> tuple[int, int]:
    """Exact deterministic integer oracle for one prime's ``loop2 -> run_prime -> unloop2``
    under Power-of-2 Dyadic Fixed-Point CRT (``acc <- (acc + sum v_dyad) mod 2^f``).
    """
    p = prob.periods[prime_idx]
    g = prob.generators[prime_idx]
    inv_q, q_mod_n = prob.crt_pairs[prime_idx]
    D = spec.center_bits
    pbits = p.bit_length()
    u_val = dlog_sum - offset_b
    max_u = (1 << (D - pbits)) * (p - 1)
    if not (0 <= u_val < max_u):
        raise ValueError(f"Input outside L1 shifted_unsigned domain at prime {prime_idx}: u={u_val}, max={max_u}")
    r_exp = u_val % (p - 1)

    b0_bits, stages, term_stage = resolve_prime_stages(pbits, spec)
    word = pow(g, (r_exp & ((1 << b0_bits) - 1)) + offset_b, p)
    word_bits = pbits
    r_width = residue_width_for(pbits, spec.source_window)
    for sel_off, sel_w in stages:
        sel_val = (r_exp >> sel_off) & ((1 << sel_w) - 1)
        mult = pow(g, sel_val << sel_off, p)
        nxt = 0
        for src_off in range(0, word_bits, spec.source_window):
            sb = min(spec.source_window, word_bits - src_off)
            chunk = (word >> src_off) & ((1 << sb) - 1)
            nxt += (mult * (chunk << src_off)) % p
        assert 0 <= nxt < (1 << r_width)
        word = nxt
        word_bits = r_width

    acc_mod = 1 << prob.len_accumulator
    delta_z = 0
    if term_stage is not None:
        sel_off, sel_w = term_stage
        sel_val = (r_exp >> sel_off) & ((1 << sel_w) - 1)
        mult = pow(g, sel_val << sel_off, p)
        for src_off in range(0, word_bits, spec.source_window):
            sb = min(spec.source_window, word_bits - src_off)
            chunk = (word >> src_off) & ((1 << sb) - 1)
            term = (mult * (chunk << src_off)) % p
            v_dyad = dyadic_crt_entry_value(term, p, inv_q, q_mod_n, prob.modulus, prob.len_accumulator)
            delta_z = (delta_z + v_dyad) % acc_mod
    else:
        for src_off in range(0, word_bits, 5):
            sb = min(5, word_bits - src_off)
            chunk = (word >> src_off) & ((1 << sb) - 1)
            term = chunk << src_off
            v_dyad = dyadic_crt_entry_value(term, p, inv_q, q_mod_n, prob.modulus, prob.len_accumulator)
            delta_z = (delta_z + v_dyad) % acc_mod

    return u_val, delta_z


def batch_segment_spans(num_primes: int, share: int, batches_per_segment: int) -> list[tuple[int, int, bool]]:
    total_batches = (num_primes + share - 1) // share
    spans = []
    for b_start in range(0, total_batches, batches_per_segment):
        b_stop = min(b_start + batches_per_segment, total_batches)
        spans.append((b_start, b_stop, b_stop == total_batches))
    return spans


def propagate_combined_cases(
    prob: PublicRNSProblem,
    spec: CombinedScheduleSpec,
    spans: Sequence[tuple[int, int, bool]],
    initial_pairs: Sequence[tuple[int, int]],
    dlog_rows: dict[int, np.ndarray] | None = None,
) -> list[list[tuple[tuple[int, ...], tuple[int, ...]]]]:
    k = spec.share
    n_shots = len(initial_pairs)
    acc_mod = 1 << prob.len_accumulator
    exps = [int(e) for e, _ in initial_pairs]
    accs = [int(z) & (acc_mod - 1) for _, z in initial_pairs]
    lanes = [[0] * k for _ in range(n_shots)]

    win_addrs_arr = np.array(
        [[(e >> (w * prob.window1)) & ((1 << prob.window1) - 1) for w in range(prob.num_windows1)] for e in exps],
        dtype=np.intp,
    )
    w_grid = np.arange(prob.num_windows1, dtype=np.intp)[None, :]

    def get_row(p_idx: int) -> np.ndarray:
        if dlog_rows is not None and p_idx in dlog_rows:
            return np.array(dlog_rows[p_idx], dtype=np.uint64, copy=True)
        return np.asarray(compute_prime_dlog_row(prob, p_idx), dtype=np.uint64)

    per_seg = []
    for b_start, b_stop, include_clear in spans:
        seg_in = [(exps[s], accs[s], *lanes[s]) for s in range(n_shots)]
        for b_idx in range(b_start, b_stop):
            indices = [
                b_idx * k + lane if (b_idx * k + lane) < len(prob.periods) else None
                for lane in range(k)
            ]
            for lane, p_idx in enumerate(indices):
                if p_idx is None:
                    continue
                row = get_row(p_idx)
                off_b = startup_offset_for_row(row, prob.periods[p_idx], prob, spec)
                d_sums = row[w_grid, win_addrs_arr].sum(axis=1).tolist()
                for s in range(n_shots):
                    u_val, dz = oracle_prime_dyadic_delta(prob, p_idx, int(d_sums[s]), off_b, spec)
                    lanes[s][lane] = u_val
                    accs[s] = (accs[s] + dz) & (acc_mod - 1)
        if include_clear:
            for s in range(n_shots):
                lanes[s] = [0] * k
        seg_out = [(exps[s], accs[s], *lanes[s]) for s in range(n_shots)]
        per_seg.append(list(zip(seg_in, seg_out)))
    return per_seg


class _PhaseCollector:
    def __init__(self, phase_bits: int):
        self.phase_bits = phase_bits
        self.size = 1 << phase_bits
        self.outcomes: list = []
        self.words = np.zeros((self.size, 1), dtype=np.uint64)

    def _ensure_bit_capacity(self, total_bits: int) -> None:
        needed_words = max(1, (total_bits + 63) // 64)
        if needed_words > self.words.shape[1]:
            pad = np.zeros((self.size, needed_words - self.words.shape[1]), dtype=np.uint64)
            self.words = np.concatenate([self.words, pad], axis=1)

    def add_direct_values(self, outs: Sequence, values_per_paddr: np.ndarray) -> None:
        m = len(outs)
        if m == 0:
            return
        if m > 64:
            for chunk_start in range(0, m, 64):
                sub_outs = outs[chunk_start : chunk_start + 64]
                self.add_direct_values(sub_outs, values_per_paddr >> chunk_start)
            return
        bit_pos = len(self.outcomes)
        self.outcomes.extend(outs)
        self._ensure_bit_capacity(len(self.outcomes))
        w_idx = bit_pos >> 6
        shift = bit_pos & 63
        vals = np.asarray(values_per_paddr, dtype=np.uint64) & np.uint64((1 << m) - 1 if m < 64 else 0xFFFFFFFFFFFFFFFF)
        self.words[:, w_idx] ^= (vals << np.uint64(shift))
        if shift + m > 64:
            self.words[:, w_idx + 1] ^= (vals >> np.uint64(64 - shift))

    def add_indexed_table(self, outs: Sequence, table_masks: Sequence[int], addr_per_paddr: np.ndarray) -> None:
        m = len(outs)
        if m == 0:
            return
        for chunk_start in range(0, m, 64):
            sub_outs = outs[chunk_start : chunk_start + 64]
            mask64 = (1 << len(sub_outs)) - 1
            arr = np.array([(int(v) >> chunk_start) & mask64 for v in table_masks], dtype=np.uint64)
            self.add_direct_values(sub_outs, arr[addr_per_paddr])


def _emit_phase_from_collector(b: Builder, address: Sequence[int], collector: _PhaseCollector, *, n1_override: int | None = None) -> None:
    if len(collector.outcomes) == 0 or not np.any(collector.words):
        for m in collector.outcomes:
            b.retire(m)
        return
    emit_universal_cz_phaseup(b, address, collector.words, collector.outcomes, n1_override=n1_override)
    for m in collector.outcomes:
        b.retire(m)


def _emit_dyadic_crt_prime(
    b: Builder,
    accumulator: Sequence[int],
    residue: Sequence[int],
    selector: Sequence[int],
    prob: PublicRNSProblem,
    prime_idx: int,
    sel_off: int,
    spec: CombinedScheduleSpec,
) -> None:
    """Bilinear shared-selector zero-phase Dyadic Mod-``2^f`` CRT accumulation for all slices of one prime."""
    p = prob.periods[prime_idx]
    g = prob.generators[prime_idx]
    inv_q, q_mod_n = prob.crt_pairs[prime_idx]
    n_acc = len(accumulator)
    n_sel = len(selector)
    sel_map, sel_trail = materialize_anf_subset_tree(b, selector)

    for src_off in range(0, len(residue), spec.source_window):
        src_slice = list(residue[src_off : src_off + spec.source_window])
        sb = len(src_slice)
        raw_terms = positive_table(p, g, sel_off, src_off, n_sel, sb)
        crt_vals = [dyadic_crt_entry_value(t, p, inv_q, q_mod_n, prob.modulus, n_acc) for t in raw_terms]
        if not any(crt_vals):
            continue
        tmp = b.qubits(n_acc)
        anf_list, low_map, low_trail = emit_bilinear_qrom_load(
            b, src_slice, sel_map, n_sel, crt_vals, tmp, keep_low_map=True
        )
        if spec.q_safe_k1:
            emit_cuccaro_add(b, accumulator, tmp, subtract=False)
        else:
            emit_qq_add(b, accumulator, tmp, subtract=False)
        outs = [b.hmr(q) for q in tmp]
        for q in reversed(tmp):
            b.recycle(q)
        anf_words = words_from_int_masks(anf_list, n_acc)
        emit_bilinear_zero_phase_cleanup(b, anf_words, sb, n_sel, low_map, sel_map, outs)
        release_anf_subset_tree(b, low_trail)

    release_anf_subset_tree(b, sel_trail)


def _emit_run_prime_combined(
    b: Builder,
    prob: PublicRNSProblem,
    dlog: Sequence[int],
    accumulator: Sequence[int],
    prime_idx: int,
    offset_b: int,
    spec: CombinedScheduleSpec,
) -> None:
    p = prob.periods[prime_idx]
    g = prob.generators[prime_idx]
    pbits = p.bit_length()
    r_width = residue_width_for(pbits, spec.source_window)
    b0_bits, stages, term_stage = resolve_prime_stages(len(dlog), spec)
    cut = min(max(0, spec.cutoff), len(stages))
    phase_bits = b0_bits + sum(sw for _, sw in stages[:cut])
    collector = _PhaseCollector(phase_bits)
    paddrs = np.arange(1 << phase_bits, dtype=np.uint64)

    # 1. Startup lookup on dlog[:b0_bits]
    startup_tab = [pow(g, a + offset_b, p) for a in range(1 << b0_bits)]
    if spec.startup_lam > 1 and (1 << b0_bits) >= spec.startup_lam:
        residue, meas_start = emit_selectswap_shared_sel(
            b, dlog[:b0_bits], [], None, startup_tab, pbits, spec.startup_lam
        )
        outs_s, masks_s = meas_start(include_bank0=False)
        collector.add_indexed_table(outs_s, masks_s, (paddrs & np.uint64((1 << b0_bits) - 1)).astype(np.int64))
    else:
        residue = b.qubits(pbits)
        emit_anf_qrom(b, dlog[:b0_bits], startup_tab, residue)

    cls_word = np.asarray(startup_tab, dtype=np.uint64)[paddrs & np.uint64((1 << b0_bits) - 1)]

    history = []
    for s_idx, (sel_off, sel_w) in enumerate(stages):
        selector = list(dlog[sel_off : sel_off + sel_w])
        sel_map, sel_trail = materialize_anf_subset_tree(b, selector)
        target = b.qubits(r_width)
        stage_terms = []
        is_skipped = s_idx < cut
        if is_skipped:
            sel_vals = (paddrs >> np.uint64(sel_off)) & np.uint64((1 << sel_w) - 1)
            next_cls_word = np.zeros_like(cls_word)

        for term_idx, src_off in enumerate(range(0, len(residue), spec.source_window)):
            src_slice = list(residue[src_off : src_off + spec.source_window])
            sb = len(src_slice)
            ab = len(selector)
            vals = positive_table(p, g, sel_off, src_off, ab, sb)
            active_w = max(1, ((term_idx + 1) * (p - 1)).bit_length()) if spec.partial_width else r_width
            val_w = min(active_w, max(1, max(vals).bit_length()))

            if is_skipped:
                low_vals = (cls_word >> np.uint64(src_off)) & np.uint64((1 << sb) - 1)
                sub_addrs = ((sel_vals << np.uint64(sb)) | low_vals).astype(np.int64)
                next_cls_word += np.asarray(vals, dtype=np.uint64)[sub_addrs]

            use_lam = spec.stage_lam if (not spec.q_safe_k1 and spec.stage_lam > 1 and (1 << (sb + ab)) >= 64) else 1
            if src_off == 0:
                if use_lam > 1:
                    bank0, meas_fn = emit_selectswap_shared_sel(b, src_slice, selector, sel_map, vals, val_w, use_lam)
                    for bit_i in range(val_w):
                        b.swap(bank0[bit_i], target[bit_i])
                    outs_e, masks_e = meas_fn(include_bank0=True)
                    mask_hi_only = [m_val >> val_w for m_val in masks_e]
                    for z_out in outs_e[:val_w]:
                        b.retire(z_out)
                    if is_skipped:
                        collector.add_indexed_table(outs_e[val_w:], mask_hi_only, sub_addrs)
                        stage_terms.append((src_off, sb, vals, val_w, None, None))
                    else:
                        stage_terms.append((src_off, sb, vals, val_w, outs_e[val_w:], mask_hi_only))
                else:
                    emit_bilinear_qrom_load(b, src_slice, sel_map, ab, vals, target[:val_w], keep_low_map=False)
                    stage_terms.append((src_off, sb, vals, val_w, None, None))
            else:
                if use_lam > 1:
                    bank0, meas_fn = emit_selectswap_shared_sel(b, src_slice, selector, sel_map, vals, val_w, use_lam)
                    emit_qq_add(b, target[:active_w], bank0)
                    outs_t, masks_t = meas_fn(include_bank0=True)
                else:
                    pay = b.qubits(val_w)
                    emit_bilinear_qrom_load(b, src_slice, sel_map, ab, vals, pay, keep_low_map=False)
                    if spec.q_safe_k1:
                        emit_cuccaro_add(b, target[:active_w], pay)
                    else:
                        emit_qq_add(b, target[:active_w], pay)
                    outs_t = [b.hmr(q) for q in pay]
                    for q in reversed(pay):
                        b.recycle(q)
                    masks_t = vals
                if is_skipped:
                    collector.add_indexed_table(outs_t, masks_t, sub_addrs)
                    stage_terms.append((src_off, sb, vals, val_w, None, None))
                else:
                    stage_terms.append((src_off, sb, vals, active_w, outs_t, masks_t))

        release_anf_subset_tree(b, sel_trail)

        if is_skipped:
            old_outs = [b.hmr(q) for q in residue]
            for q in reversed(residue):
                b.recycle(q)
            collector.add_direct_values(old_outs, cls_word)
            cls_word = next_cls_word
        else:
            history.append((sel_off, sel_w, residue, stage_terms))
        residue = target

    # 2. Power-of-2 Dyadic CRT accumulation
    if term_stage is not None:
        sel_off, sel_w = term_stage
        selector = list(dlog[sel_off : sel_off + sel_w])
        _emit_dyadic_crt_prime(b, accumulator, residue, selector, prob, prime_idx, sel_off, spec)
    else:
        for src_off in range(0, len(residue), 5):
            src_slice = list(residue[src_off : src_off + 5])
            inv_q, q_mod_n = prob.crt_pairs[prime_idx]
            crt_vals = [
                dyadic_crt_entry_value(x << src_off, p, inv_q, q_mod_n, prob.modulus, len(accumulator))
                for x in range(1 << len(src_slice))
            ]
            if any(crt_vals):
                tmp = b.qubits(len(accumulator))
                anf_list, low_map, low_trail = emit_bilinear_qrom_load(
                    b, src_slice, {0: None}, 0, crt_vals, tmp, keep_low_map=True
                )
                if spec.q_safe_k1:
                    emit_cuccaro_add(b, accumulator, tmp)
                else:
                    emit_qq_add(b, accumulator, tmp)
                outs = [b.hmr(q) for q in tmp]
                for q in reversed(tmp):
                    b.recycle(q)
                emit_bilinear_zero_phase_cleanup(
                    b, words_from_int_masks(anf_list, len(accumulator)), len(src_slice), 0, low_map, {0: None}, outs
                )
                release_anf_subset_tree(b, low_trail)

    # 3. Reverse uncompute any retained stages above cutoff
    for sel_off, sel_w, pred_reg, stage_terms in reversed(history):
        selector = list(dlog[sel_off : sel_off + sel_w])
        sel_map, sel_trail = materialize_anf_subset_tree(b, selector)
        for src_off, sb, vals, active_or_val_w, fwd_outs, fwd_masks in reversed(stage_terms):
            src_slice = list(pred_reg[src_off : src_off + sb])
            addr_wires = src_slice + selector
            N_sub = 1 << len(addr_wires)
            idx_ident = np.arange(N_sub, dtype=np.int64)
            if src_off == 0:
                res_outs = [b.hmr(q) for q in residue]
                for q in reversed(residue):
                    b.recycle(q)
                # Clean up using bilinear zero-phase if q_safe_k1 is False, or UniversalAllToAllCZ!
                low_map, low_trail = materialize_anf_subset_tree(b, src_slice)
                pc = _PhaseCollector(len(addr_wires))
                pc.add_indexed_table(res_outs, vals, idx_ident)
                if fwd_outs:
                    pc.add_indexed_table(fwd_outs, fwd_masks, idx_ident)
                anf_words = mobius_transform_u64_matrix(pc.words, len(addr_wires))
                emit_bilinear_zero_phase_cleanup(b, anf_words, sb, sel_w, low_map, sel_map, pc.outcomes)
                release_anf_subset_tree(b, low_trail)
            elif spec.q_safe_k1:
                # Q-safe k=1 reverse: release low_map before Cuccaro add so temp qubits stay <= 75 (1,306 Q).
                val_w = min(active_or_val_w, max(1, max(vals).bit_length()))
                pay = b.qubits(val_w)
                emit_bilinear_qrom_load(b, src_slice, sel_map, sel_w, vals, pay, keep_low_map=False)
                emit_cuccaro_add(b, residue[:active_or_val_w], pay, subtract=True)
                rev_outs = [b.hmr(q) for q in pay]
                for q in reversed(pay):
                    b.recycle(q)
                pc = _PhaseCollector(len(addr_wires))
                if fwd_outs:
                    pc.add_indexed_table(fwd_outs, fwd_masks, idx_ident)
                pc.add_indexed_table(rev_outs, vals, idx_ident)
                _emit_phase_from_collector(b, addr_wires, pc)
            else:
                # Zero-phase bilinear reverse uncompute (0 phase-lookup Toffolis!)
                val_w = min(active_or_val_w, max(1, max(vals).bit_length()))
                pay = b.qubits(val_w)
                _, low_map, low_trail = emit_bilinear_qrom_load(
                    b, src_slice, sel_map, sel_w, vals, pay, keep_low_map=True
                )
                emit_qq_add(b, residue[:active_or_val_w], pay, subtract=True)
                rev_outs = [b.hmr(q) for q in pay]
                for q in reversed(pay):
                    b.recycle(q)
                pc = _PhaseCollector(len(addr_wires))
                if fwd_outs:
                    pc.add_indexed_table(fwd_outs, fwd_masks, idx_ident)
                pc.add_indexed_table(rev_outs, vals, idx_ident)
                anf_words = mobius_transform_u64_matrix(pc.words, len(addr_wires))
                emit_bilinear_zero_phase_cleanup(b, anf_words, sb, sel_w, low_map, sel_map, pc.outcomes)
                release_anf_subset_tree(b, low_trail)
        release_anf_subset_tree(b, sel_trail)
        residue = pred_reg

    # 4. Demolish boundary residue and emit prefix phase table
    boundary_outs = [b.hmr(q) for q in residue]
    for q in reversed(residue):
        b.recycle(q)
    collector.add_direct_values(boundary_outs, cls_word)
    _emit_phase_from_collector(b, dlog[:phase_bits], collector, n1_override=spec.phase_n1_override)


def _emit_combined_loop1_update(
    b: Builder,
    prob: PublicRNSProblem,
    exponent: Sequence[int],
    lanes: Sequence[Sequence[int]],
    old_indices: Sequence[int | None],
    new_indices: Sequence[int | None],
    rows_cache: dict[int, np.ndarray],
    window_collectors: list[_PhaseCollector] | None,
    spec: CombinedScheduleSpec,
) -> None:
    k = spec.share
    narrow = (max(prob.periods) - 1).bit_length() + 1  # 21 bits for 20-bit primes
    word_mask = (1 << narrow) - 1
    active_lanes = [lane for lane in range(k) if not (old_indices[lane] is None and new_indices[lane] is None)]
    if not active_lanes:
        return
    rows_py: dict[int, list[list[int]]] = {
        p_idx: rows_cache[p_idx].tolist()
        for p_idx in {p for p in (*old_indices, *new_indices) if p is not None}
    }

    if spec.loop1_mode == "rolling_anf":
        for w_idx in range(prob.num_windows1):
            addr = list(exponent[w_idx * prob.window1 :][: prob.window1])
            n_w = len(addr)
            cnt = 1 << n_w
            packed_diffs = [0] * cnt
            nonempty_slots: list[tuple[int, int]] = []
            for slot_i, lane in enumerate(active_lanes):
                old_p, new_p = old_indices[lane], new_indices[lane]
                old_row = None if old_p is None else rows_py[old_p][w_idx]
                new_row = None if new_p is None else rows_py[new_p][w_idx]
                shift_bits = narrow * slot_i
                any_nz = False
                for av in range(cnt):
                    d_val = ((0 if new_row is None else new_row[av]) - (0 if old_row is None else old_row[av])) & word_mask
                    if d_val:
                        packed_diffs[av] |= d_val << shift_bits
                        any_nz = True
                if any_nz:
                    nonempty_slots.append((slot_i, lane))
            if not nonempty_slots:
                continue

            m2w, trail = materialize_anf_subset_tree(b, addr)
            packed_anf = mobius_transform_int_list(packed_diffs, n_w)
            all_outs = []
            used_packed_anf = [0] * cnt
            for out_idx, (slot_i, lane) in enumerate(nonempty_slots):
                src_shift = narrow * slot_i
                dst_shift = narrow * out_idx
                anf_lane = [(packed_anf[mask] >> src_shift) & word_mask for mask in range(cnt)]
                for mask, lv in enumerate(anf_lane):
                    if lv:
                        used_packed_anf[mask] |= lv << dst_shift
                word_reg = b.qubits(narrow)
                fast_fanout_anf_lane(b, m2w, anf_lane, word_reg)
                emit_signed_packed_lane_add(b, lanes[lane], word_reg)
                outs_l = [b.hmr(q) for q in word_reg]
                for q in reversed(word_reg):
                    b.recycle(q)
                all_outs.extend(outs_l)

            anf_words = words_from_int_masks(used_packed_anf, len(all_outs))
            emit_fast_z_mask_cleanup(b, m2w, anf_words, all_outs, cnt)
            for m in all_outs:
                b.retire(m)
            release_anf_subset_tree(b, trail)

    elif spec.loop1_mode == "bilinear_rolling_anf":
        # Q-safe zero-phase Loop 1 for k=1 (uses <= 53 temporary qubits at w1=6, 0 final_phase!)
        for w_idx in range(prob.num_windows1):
            addr = list(exponent[w_idx * prob.window1 :][: prob.window1])
            n_w = len(addr)
            cnt = 1 << n_w
            n_lo = n_w >> 1
            n_hi = n_w - n_lo
            hi_map, hi_trail = materialize_anf_subset_tree(b, addr[n_lo:])
            for lane in active_lanes:
                old_p, new_p = old_indices[lane], new_indices[lane]
                old_row = None if old_p is None else rows_py[old_p][w_idx]
                new_row = None if new_p is None else rows_py[new_p][w_idx]
                diffs = [
                    ((0 if new_row is None else new_row[av]) - (0 if old_row is None else old_row[av])) & word_mask
                    for av in range(cnt)
                ]
                if not any(diffs):
                    continue
                word_reg = b.qubits(narrow)
                anf_l, lo_map, lo_trail = emit_bilinear_qrom_load(
                    b, addr[:n_lo], hi_map, n_hi, diffs, word_reg, keep_low_map=True
                )
                emit_signed_packed_lane_add(b, lanes[lane], word_reg)
                outs_l = [b.hmr(q) for q in word_reg]
                for q in reversed(word_reg):
                    b.recycle(q)
                emit_bilinear_zero_phase_cleanup(
                    b, words_from_int_masks(anf_l, narrow), n_lo, n_hi, lo_map, hi_map, outs_l
                )
                release_anf_subset_tree(b, lo_trail)
            release_anf_subset_tree(b, hi_trail)

    elif spec.loop1_mode == "packed_zero_tail":
        assert window_collectors is not None
        m_active = len(active_lanes)
        for w_idx in range(prob.num_windows1):
            addr = list(exponent[w_idx * prob.window1 :][: prob.window1])
            cnt = 1 << len(addr)
            packed = [0] * cnt
            for slot_i, lane in enumerate(active_lanes):
                old_p, new_p = old_indices[lane], new_indices[lane]
                old_row = None if old_p is None else rows_py[old_p][w_idx]
                new_row = None if new_p is None else rows_py[new_p][w_idx]
                shift_bits = narrow * slot_i
                for av in range(cnt):
                    before = 0 if old_row is None else old_row[av]
                    after = 0 if new_row is None else new_row[av]
                    packed[av] |= ((after - before) & word_mask) << shift_bits
            output = b.qubits(narrow * m_active)
            emit_anf_qrom(b, addr, packed, output)
            for slot_i, lane in enumerate(active_lanes):
                emit_signed_packed_lane_add(b, lanes[lane], output[slot_i * narrow : (slot_i + 1) * narrow])
            outs = [b.hmr(q) for q in output]
            for q in reversed(output):
                b.recycle(q)
            window_collectors[w_idx].add_indexed_table(outs, packed, np.arange(cnt, dtype=np.int64))
    else:
        raise ValueError(f"Unknown loop1_mode: {spec.loop1_mode}")

    for lane in active_lanes:
        old_p, new_p = old_indices[lane], new_indices[lane]
        old_off = 0 if old_p is None else startup_offset_for_row(rows_cache[old_p], prob.periods[old_p], prob, spec)
        new_off = 0 if new_p is None else startup_offset_for_row(rows_cache[new_p], prob.periods[new_p], prob, spec)
        delta_off = new_off - old_off
        if delta_off != 0:
            emit_const_add(b, lanes[lane], delta_off, subtract=True)


def build_combined_segment(
    prob: PublicRNSProblem,
    spec: CombinedScheduleSpec,
    batch_start: int,
    batch_stop: int,
    *,
    include_final_clear: bool,
    dlog_rows: dict[int, np.ndarray] | None = None,
) -> tuple[Builder, dict]:
    k = spec.share
    b = Builder(recycle_outcomes=True)
    exponent = b.qubits(prob.num_input_qubits)
    accumulator = b.qubits(prob.len_accumulator)
    lanes = [b.qubits(spec.center_bits) for _ in range(k)]
    b.qubit_register(exponent)
    b.qubit_register(accumulator)
    for lane_reg in lanes:
        b.qubit_register(lane_reg)

    needed_primes: set[int] = set()
    if batch_start > 0:
        for lane in range(k):
            p_prev = (batch_start - 1) * k + lane
            if p_prev < len(prob.periods):
                needed_primes.add(p_prev)
    for b_idx in range(batch_start, batch_stop):
        for lane in range(k):
            p_cur = b_idx * k + lane
            if p_cur < len(prob.periods):
                needed_primes.add(p_cur)

    rows_cache: dict[int, np.ndarray] = {}
    for p_idx in sorted(needed_primes):
        rows_cache[p_idx] = (
            np.array(dlog_rows[p_idx], dtype=np.uint32, copy=True)
            if (dlog_rows is not None and p_idx in dlog_rows)
            else compute_prime_dlog_row(prob, p_idx)
        )

    window_collectors = (
        [_PhaseCollector(len(exponent[w * prob.window1 :][: prob.window1])) for w in range(prob.num_windows1)]
        if spec.loop1_mode == "packed_zero_tail"
        else None
    )

    cur_indices: list[int | None] = (
        [(batch_start - 1) * k + lane if ((batch_start - 1) * k + lane) < len(prob.periods) else None for lane in range(k)]
        if batch_start > 0
        else [None] * k
    )

    primes_processed = 0
    for b_idx in range(batch_start, batch_stop):
        next_indices = [b_idx * k + lane if (b_idx * k + lane) < len(prob.periods) else None for lane in range(k)]
        b.set_phase("loop1")
        _emit_combined_loop1_update(b, prob, exponent, lanes, cur_indices, next_indices, rows_cache, window_collectors, spec)
        cur_indices = next_indices

        for lane, p_idx in enumerate(cur_indices):
            if p_idx is None:
                continue
            b.set_phase("prime_kernel")
            p = prob.periods[p_idx]
            dlog = emit_fast_nonrestoring_loop2(b, lanes[lane], p - 1, p.bit_length(), inverse=False)
            off_b = startup_offset_for_row(rows_cache[p_idx], p, prob, spec)
            _emit_run_prime_combined(b, prob, dlog, accumulator, p_idx, off_b, spec)
            emit_fast_nonrestoring_loop2(b, lanes[lane], p - 1, p.bit_length(), inverse=True)
            primes_processed += 1

    if include_final_clear:
        b.set_phase("loop1_final")
        _emit_combined_loop1_update(b, prob, exponent, lanes, cur_indices, [None] * k, rows_cache, window_collectors, spec)

    b.set_phase("final_phase")
    if window_collectors is not None:
        for w_idx, col in enumerate(window_collectors):
            addr = list(exponent[w_idx * prob.window1 :][: prob.window1])
            _emit_phase_from_collector(b, addr, col)

    meta = {
        "variant_id": spec.variant_id,
        "problem": prob.name,
        "batch_start": batch_start,
        "batch_stop": batch_stop,
        "include_final_clear": include_final_clear,
        "primes_processed": primes_processed,
        "phase_summary": b.phase_summary(),
        "next_out": b.next_out,
        "peak_qubits": b.peak_qubits,
        "spec": asdict(spec),
    }
    return b, meta


def compute_l1_dyadic_certificate(
    prob: PublicRNSProblem,
    spec: CombinedScheduleSpec,
    *,
    trace_upper: Fraction | float | None = None,
    trace_source: str | None = None,
) -> dict:
    """Exact ``fractions.Fraction`` Proposition 3 coset-mask error-diameter Eq. (A5) certificate
    under BOTH:
    1. Declared worst-case ``min_wraparound_gap`` (diameter ``1*2^-f + 2^-g_gap``);
    2. Exact pinned RNS product deviation ``min(L % N, -L % N) / N``.

    ``trace_upper`` should be the per-shot trace-distance bound ``eta >= 2*sqrt(p_bad)`` certified
    for this exact prime family and center width ``D``. If omitted, computes the rigorous analytic
    Eq. (C5) Hoeffding bound for offset-unsigned ``D`` directly from ``prob.periods``.
    """
    f = prob.len_accumulator
    mask_b = prob.mask_bits
    gap_b = prob.min_wraparound_gap
    shots_nom = spec.s_param + 1
    if spec.terminal_crt:
        slices_per_prime = (residue_width_for(prob.prime_bits, spec.source_window) + spec.source_window - 1) // spec.source_window
    else:
        slices_per_prime = (prob.prime_bits + 4) // 5
    m_crt = len(prob.periods) * slices_per_prime
    mask_prop = Fraction(1, 1 << (f - mask_b))

    if trace_upper is None:
        from mgf_certificates.error_budget import shifted_unsigned_hoeffding_certificate
        h_cert = shifted_unsigned_hoeffding_certificate(
            prob.periods, windows=prob.num_windows1, bits=spec.center_bits, bin_width=256
        )
        num_s, den_s = h_cert["trace_fraction"]
        trace_frac = Fraction(int(num_s), int(den_s))
        resolved_source = trace_source or "analytic_hoeffding_eq_C5_bin256"
    elif isinstance(trace_upper, Fraction):
        trace_frac = trace_upper
        resolved_source = trace_source or "caller_exact_fraction"
    else:
        trace_frac = Fraction(str(trace_upper))
        resolved_source = trace_source or "caller_supplied_eta"

    # 1. Declared gap bound: Proposition 3 Eq. (A5) rigorous diameter (1*2^-f + 2^-g_gap)
    #    plus single-point worst-case deviation (0.5*2^-f + 2^-g_gap) for reference
    eps_odd_gap = m_crt * (Fraction(2, 1 << f) + Fraction(1, 1 << gap_b))
    eps_dyadic_diam_gap = m_crt * (Fraction(1, 1 << f) + Fraction(1, 1 << gap_b))
    eps_dyadic_sp_gap = m_crt * (Fraction(1, 2 * (1 << f)) + Fraction(1, 1 << gap_b))
    keep_odd_gap = (1 - mask_prop - eps_odd_gap / mask_prop) - shots_nom * trace_frac
    keep_dyadic_diam_gap = (1 - mask_prop - eps_dyadic_diam_gap / mask_prop) - shots_nom * trace_frac
    keep_dyadic_sp_gap = (1 - mask_prop - eps_dyadic_sp_gap / mask_prop) - shots_nom * trace_frac

    # 2. Exact pinned RNS deviation bound (Eq. (A5) diameter: 1*2^-f + |delta|/N)
    dev = min(prob.full_rns_product % prob.modulus, (-prob.full_rns_product) % prob.modulus)
    eps_dyadic_diam_pinned = m_crt * (Fraction(1, 1 << f) + Fraction(dev, prob.modulus))
    eps_dyadic_sp_pinned = m_crt * (Fraction(1, 2 * (1 << f)) + Fraction(dev, prob.modulus))
    keep_dyadic_diam_pinned = (1 - mask_prop - eps_dyadic_diam_pinned / mask_prop) - shots_nom * trace_frac

    if keep_dyadic_diam_gap <= 0 or keep_dyadic_diam_pinned <= 0:
        raise ValueError(
            f"Certified retention probability is non-positive for {spec.variant_id} "
            f"(declared keep={float(keep_dyadic_diam_gap):.6f}, pinned keep={float(keep_dyadic_diam_pinned):.6f}) "
            f"under trace_source={resolved_source!r} (eta={float(trace_frac):.3e}). "
            f"For compressed lane width D={spec.center_bits} where analytic Hoeffding saturates, "
            f"pass an explicit MGF certificate via trace_upper=..."
        )

    fac_odd_gap = (Fraction(shots_nom, 1) / keep_odd_gap) if keep_odd_gap > 0 else None
    fac_dyadic_diam_gap = Fraction(shots_nom, 1) / keep_dyadic_diam_gap
    fac_dyadic_sp_gap = Fraction(shots_nom, 1) / keep_dyadic_sp_gap
    fac_dyadic_diam_pinned = Fraction(shots_nom, 1) / keep_dyadic_diam_pinned

    return {
        "schema": "combined-l1-dyadic-crt-rational-certificate-v3-rigorous-diameter",
        "variant_id": spec.variant_id,
        "s": spec.s_param,
        "nominal_shots": shots_nom,
        "prime_count": len(prob.periods),
        "slices_per_prime": slices_per_prime,
        "M_CRT_terms": m_crt,
        "len_accumulator": f,
        "mask_bits": mask_b,
        "declared_min_wraparound_gap": gap_b,
        "pinned_rns_effective_gap_bits": math.log2(prob.modulus / dev) if dev > 0 else float("inf"),
        "trace_source": resolved_source,
        "trace_fraction_num_den": [str(trace_frac.numerator), str(trace_frac.denominator)],
        "trace_upper_float": float(trace_frac),
        "dyadic_diameter_formula": "1*2^-f + 2^-g_gap (Proposition 3, Eq. (A5))",
        "declared_gap_bound": {
            "epsilon_odd_crt": float(eps_odd_gap),
            "epsilon_dyadic_diameter": float(eps_dyadic_diam_gap),
            "epsilon_dyadic_single_point": float(eps_dyadic_sp_gap),
            "keep_odd_crt": float(keep_odd_gap),
            "keep_dyadic_diameter": float(keep_dyadic_diam_gap),
            "keep_dyadic_single_point": float(keep_dyadic_sp_gap),
            "factor_per_shot_odd_crt": float(fac_odd_gap) if fac_odd_gap is not None else None,
            "factor_per_shot_dyadic_diameter": float(fac_dyadic_diam_gap),
            "factor_per_shot_dyadic_single_point": float(fac_dyadic_sp_gap),
        },
        "pinned_rns_exact_bound": {
            "modular_deviation_over_N": float(Fraction(dev, prob.modulus)),
            "epsilon_dyadic_diameter_exact": float(eps_dyadic_diam_pinned),
            "epsilon_dyadic_single_point_exact": float(eps_dyadic_sp_pinned),
            "keep_dyadic_diameter_exact": float(keep_dyadic_diam_pinned),
            "factor_per_shot_dyadic_diameter_exact": float(fac_dyadic_diam_pinned),
            "epsilon_dyadic_diameter_exact_num_den": [
                str(eps_dyadic_diam_pinned.numerator),
                str(eps_dyadic_diam_pinned.denominator),
            ],
        },
    }
