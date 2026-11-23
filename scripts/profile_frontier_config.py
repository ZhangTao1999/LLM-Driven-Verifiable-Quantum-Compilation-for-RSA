#!/usr/bin/env python3
"""Compile and profile any RSA-2048 frontier configuration directly from data/rns_inputs/.

Example:
    python scripts/profile_frontier_config.py --id S4_K22_W8_CMB
    python scripts/profile_frontier_config.py --id S8_K3_W8_OPT
    python scripts/profile_frontier_config.py --id S4_K26_W8_EXT
    python scripts/profile_frontier_config.py --list
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

if sys.version_info < (3, 10):
    raise SystemExit("Python >= 3.10 is required to run this profiling script.")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from flat_stream import Builder
import compiler.bridge as cfb


def spec_from_record(rec: dict) -> cfb.CombinedScheduleSpec:
    """Deserialize the exact CombinedScheduleSpec stored alongside each frontier record."""
    if "spec" not in rec:
        raise KeyError(f"Record {rec.get('id')!r} is missing its embedded 'spec' dictionary.")
    sp = dict(rec["spec"])
    sp["stage_widths"] = tuple(int(x) for x in sp["stage_widths"])
    return cfb.CombinedScheduleSpec(**sp)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile & profile an RSA-2048 frontier configuration.")
    ap.add_argument(
        "--id",
        type=str,
        default="S4_K22_W8_CMB",
        help="Configuration ID from data/combined_frontier_49_configs.json or data/hardware_and_kext_table11.json",
    )
    ap.add_argument("--list", action="store_true", help="List all 30 Pareto-optimal and 5 k>24 extension configuration IDs")
    args = ap.parse_args()

    master = json.loads((REPO / "data/combined_frontier_49_configs.json").read_text(encoding="utf-8"))
    by = {r["id"]: r for r in master["all_configs"]}
    t11_path = REPO / "data/hardware_and_kext_table11.json"
    ext_rows = []
    if t11_path.exists():
        t11 = json.loads(t11_path.read_text(encoding="utf-8"))
        ext_rows = t11.get("rolling_anf_k_extension", [])
        for r in ext_rows:
            by[r["id"]] = r
    if args.list:
        print("30 Pareto-optimal RSA-2048 configurations (strict bit-level Toffoli & matched b-2):")
        for pid in master["pareto_ids_all"]:
            r = by[pid]
            print(
                f"  {pid:22s} (s={r['s']:2d}, k={r['k']:2d}, w={r['w']}, D={r['D']}u, f={r['f']}) "
                f"Q={r['Q_peak']:4d} T_fact={r['T_fact_strict']/1e9:.4f}B (b-2: {r['T_fact_same_adder']/1e9:.4f}B)"
            )
        if ext_rows:
            print("\n5 unconstrained k > 24 rolling-ANF extension configurations (Table V(b)):")
            for r in ext_rows:
                print(
                    f"  {r['id']:22s} (s={r['s']:2d}, k={r['k']:2d}, w={r['w']}, D={r['D']}u, f={r['f']}) "
                    f"Q={r['Q_peak']:4d} T_fact={r['T_fact_strict']/1e9:.4f}B (b-2: {r['T_fact_same_adder']/1e9:.4f}B)"
                )
        return

    assert args.id in by, f"Unknown ID {args.id!r}; run with --list to see valid IDs."
    rec = by[args.id]
    spec = spec_from_record(rec)
    rns_path = REPO / "data/rns_inputs" / f"{rec['family']}.json"

    t0 = time.perf_counter()
    prob = cfb.load_public_rns_problem(rns_path, max_primes=None, name=rec["family"], spec=spec)
    P, k = len(prob.periods), spec.share
    n_batches = (P + k - 1) // k
    interior_ids = sorted({
        max(1, min(n_batches - 2, n_batches // 4)),
        max(1, min(n_batches - 2, n_batches // 2)),
        max(1, min(n_batches - 2, (3 * n_batches) // 4)),
    })

    orig_emit = Builder._emit_op
    Builder._emit_op = lambda self, op: None
    try:
        _, m0 = cfb.build_combined_segment(prob, spec, 0, 1, include_final_clear=False)
        p0 = {it["name"]: it["toffoli"] for it in m0["phase_summary"]["phases"]}
        q_peak = m0["peak_qubits"]
        int_l1, int_pk, int_p = [], [], 0
        for bid in interior_ids:
            _, mi = cfb.build_combined_segment(prob, spec, bid, bid + 1, include_final_clear=False)
            pi = {it["name"]: it["toffoli"] for it in mi["phase_summary"]["phases"]}
            int_l1.append(pi["loop1"])
            int_pk.append(pi["prime_kernel"])
            int_p += k
            q_peak = max(q_peak, mi["peak_qubits"])
        _, ml = cfb.build_combined_segment(prob, spec, n_batches - 1, n_batches, include_final_clear=True)
        pl = {it["name"]: it["toffoli"] for it in ml["phase_summary"]["phases"]}
        q_peak = max(q_peak, ml["peak_qubits"])
        n_last = P - (n_batches - 1) * k
    finally:
        Builder._emit_op = orig_emit

    tot_l1 = p0["loop1"] + max(0, n_batches - 2) * (sum(int_l1) / len(int_l1)) + pl["loop1"]
    tot_pk = p0["prime_kernel"] + (P - k - n_last) * (sum(int_pk) / int_p) + pl["prime_kernel"]
    t_shot = tot_l1 + pl.get("loop1_final", 0) + tot_pk + pl.get("final_phase", 0)
    fac = rec["budget"]["factor_declared"]
    pk_qq_nom = 0 if spec.q_safe_k1 else (28 if spec.cutoff == 3 else 35)
    qq_per_p = prob.num_windows1 + pk_qq_nom
    t_fact_strict = t_shot * fac
    t_fact_sa = (t_shot - P * qq_per_p) * fac
    dt = time.perf_counter() - t0

    print(
        f"\nProfiled {args.id} ({rec['family']}, s={spec.s_param}, k={k}, w={prob.window1}, "
        f"D={spec.center_bits}u, f={prob.len_accumulator}, b0={spec.b0}, term_sel={spec.terminal_sel_bits}) in {dt:.1f}s:"
    )
    print(f"  Q_peak               = {q_peak:,} Q  (stored: {rec['Q_peak']:,} Q)")
    print(f"  T_shot (bit-level)   = {t_shot:,.1f}  (stored: {rec['T_shot_strict']:,.1f})")
    print(f"  T_fact (bit-level)   = {t_fact_strict/1e9:.4f} x 10^9  (stored: {rec['T_fact_strict']/1e9:.4f} x 10^9)")
    print(f"  T_fact (matched b-2) = {t_fact_sa/1e9:.4f} x 10^9  (stored: {rec['T_fact_same_adder']/1e9:.4f} x 10^9)")
    print(f"  Q * T_fact           = {q_peak * t_fact_strict / 1e12:.3f} x 10^12  (b-2: {q_peak * t_fact_sa / 1e12:.3f} x 10^12)")
    assert q_peak == rec["Q_peak"], f"Q_peak mismatch: {q_peak} != {rec['Q_peak']}"
    assert qq_per_p == rec["exact_qq_adds_per_prime"], f"Nominal QQ adders mismatch: {qq_per_p} != {rec['exact_qq_adds_per_prime']}"
    if rec.get("all_prime_32shot_certified") or rec.get("profiling_mode") == "exact_all_prime_monolithic":
        if "T_shot_stratified_orig" in rec:
            strat_err = abs(t_shot - rec["T_shot_stratified_orig"]) / rec["T_shot_stratified_orig"]
            assert strat_err < 1e-6, f"Stratified T_shot mismatch: {strat_err:.3e}"
        tol = 1.5e-3  # 5-batch stratified live profile vs exact all-prime measurement (max 0.103% on S14)
    elif rec.get("profiling_mode") == "3_batch_midpoint":
        tol = 2.0e-3  # 5-batch stratified live profile vs 3-batch midpoint Table V(b) record (<= 0.11%)
    else:
        tol = 1e-6
    rel_err = abs(t_fact_strict - rec["T_fact_strict"]) / rec["T_fact_strict"]
    assert rel_err < tol, f"T_fact mismatch: rel_err={rel_err:.3e} >= tol={tol}"


if __name__ == "__main__":
    main()
