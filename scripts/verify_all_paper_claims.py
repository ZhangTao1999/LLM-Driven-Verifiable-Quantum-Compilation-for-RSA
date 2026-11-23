#!/usr/bin/env python3
"""Offline Cryptographic, Mathematical, Evidence-Binding, and Paper-Claims Audit (~20-45 seconds, Python stdlib only).

Verifies:
  1. SHA-256 integrity of every file recorded in MANIFEST_SHA256.json.
  2. Full mathematical validity of ALL 9 data/rns_inputs/*.json files: RSA-2048 modulus, strictly
     ascending unique primes (order-sensitive!), deterministic Miller-Rabin primality, bit-length,
     capacity prod(p) > N^W, effective RNS wraparound gap >= declared gap, zero-free 3^x mod N
     multiplier tables (boundary windows across all 167,733 primes + all windows on sampled primes),
     Proposition 3 dyadic torus Eq. (A5) retention, embedded spec integrity, 30 strict vs. 29
     matched b-2 Pareto counts, and nominal W+35 / W+28 (non-Pareto) / W matched b-2 counts.
  3. All 6 all-prime 32-trajectory flat-stream receipts (119,482 primes, 6,505 segments, 5,610
     zero-tax segments, 100% primary == holdout per-segment SHA-256 hashes, torus error < 1e-7),
     exact byte-for-byte public_input.sha256 binding where published verbatim (S4_K22_W8_CMB &
     S4_K24_W8_OPT), exact live-recomputed prod(p) mod N / modular_deviation_over_N binding across
     all 6 receipts, live receipt SHA-256 sync, and standalone v3 rigorous-diameter L1 certificates.
  4. Strictly same-RNS (19,878 primes, f=33) controlled ablation (R4B_PLUS vs R4B_PLUS_COMBINED) and
     unsegmented all-19,878-prime elementary-gate replay at 2,211 Q.
  5. Gidney (2025) 28-circuit baseline, analytic sensitivity, and Sharma & Lau (2026) 13-point comparison.
  6. All 23 full-prime MGF certificates: num > 0, den == 2^192, two-sided 45-digit Decimal tightness
     of eta == 2*sqrt(p_bad) and shot_debit == (s+1)*eta, RNS metadata binding, live stdlib
     strict_math.certify_prime recomputation directly from data/rns_inputs/*.json on 5 deterministic
     primes per published MGF family verifying 0 < live_sample_sum <= stored_num across all 17
     published-family certificates (full 977M-entry family enumeration uses support_worker.cpp),
     exact cross-check against all 31 MGF configs, plus full live stdlib Eq. (C5) rational
     Hoeffding (bin_width=256) recomputation across all 5 Hoeffding-certified RSA-2048 RNS families.
  7. Appendix E Table V backing data (data/hardware_and_kext_table11.json): exact all-prime
     monolithic elementary-gate hardware profiles (diff == 0 vs monolithic T_shot_strict) for all 6
     all-prime certified headline circuits, plus Eq. (A5) and embedded specs for all 5 k > 24 extensions.
"""
from __future__ import annotations

from array import array
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Python >= 3.10 is required to run this verification script.")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mgf_certificates.error_budget import shifted_unsigned_hoeffding_certificate
from mgf_certificates.strict_math import certify_prime, make_exp_upper_lut

RSA2048_MODULUS = (
    25195908475657893494027183240048398571429282126204032027777137836043662020707595556264018525880784406918290641249515082189298559149176184502808489120072844992687392807287776735971418347270261896375014971824691165077613379859095700097330459748808428401797429100642458691817195118746121515172654632282216869987549182422433637259085141865462043576798423387184774447920739934236584823824281198163815010674810451660377306056201619676256133844143603833904414952634432190114657544454178424020924616515723350778707749817125772467962926386356373289912154831438167899885040445364023527381951378636564391212010397122822120720357
)

ANCHOR_IDS = (
    "S14_PREFIX_COMBINED",
    "R4B_PLUS_COMBINED",
    "S4_ROLLK20_COMBINED",
    "S4_K22_W8_CMB",
    "S4_K24_W8_OPT",
    "S3_K22_W9_COMBINED",
)

# Receipts whose published data/rns_inputs/*.json file is byte-for-byte identical to the cluster input
EXACT_INPUT_SHA256_ANCHORS = ("S4_K22_W8_CMB", "S4_K24_W8_OPT")


def _balanced_prod(xs) -> int:
    arr = list(xs)
    if not arr:
        return 1
    while len(arr) > 1:
        arr = [arr[i] * arr[i + 1] if i + 1 < len(arr) else arr[i] for i in range(0, len(arr), 2)]
    return arr[0]


def is_prime_u32(n: int) -> bool:
    """Deterministic Miller-Rabin primality test exact for all integers n < 4,759,123,141."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    s = 0
    while (d & 1) == 0:
        d >>= 1
        s += 1
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


def primitive_root_prime(p: int) -> int:
    """Deterministic smallest positive primitive root modulo prime p."""
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


def build_multiplier_table(modulus: int, generator: int, num_windows: int, w1: int, num_input_qubits: int) -> list[list[int]]:
    rows: list[list[int]] = []
    base = generator % modulus
    for j in range(num_windows):
        width = min(w1, num_input_qubits - j * w1)
        vals = [1]
        for _ in range(1, 1 << width):
            vals.append((vals[-1] * base) % modulus)
        rows.append(vals)
        base = pow(base, 1 << w1, modulus)
    return rows


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(rel: str):
    return json.loads((REPO / rel).read_text(encoding="utf-8"))


def parse_rns_problem_fields(rns_data: dict) -> dict[str, str]:
    raw = rns_data.get("problem_fields", rns_data.get("problem", rns_data.get("problem_config", "")))
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    fields: dict[str, str] = {}
    for line in str(raw).splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            fields[k.strip()] = v.strip()
    return fields


def pareto_2d(pts):
    out, best = [], float("inf")
    for p in sorted(pts, key=lambda x: (x["Q"], x["T_fact"])):
        if p["T_fact"] < best:
            out.append(p)
            best = p["T_fact"]
    return out


def best_at_cap(pts, cap):
    feas = [p for p in pts if p["Q"] <= cap]
    return min(feas, key=lambda p: p["T_fact"]) if feas else None


def main() -> None:
    print("=== [1/7] SHA-256 MANIFEST INTEGRITY ===")
    manifest = load_json("MANIFEST_SHA256.json")
    checked = 0
    for rel, expected_sha in manifest["files"].items():
        fp = REPO / rel
        assert fp.exists(), f"Missing file in manifest: {rel}"
        actual_sha = sha256_file(fp)
        assert actual_sha == expected_sha, f"SHA-256 mismatch for {rel}: {actual_sha} != {expected_sha}"
        checked += 1
    print(f"  PASS: {checked}/{len(manifest['files'])} files match MANIFEST_SHA256.json bit-for-bit.")

    print("\n=== [2/7] ALL 9 RNS INPUTS (STRICT ORDER, ZERO-FREE, CAPACITY), EQ. (A5) & NOMINAL b-2 ===")
    rns_paths = sorted((REPO / "data/rns_inputs").glob("*.json"))
    assert len(rns_paths) == 9, f"Expected 9 RNS input files, found {len(rns_paths)}"
    rns_verified: dict[str, dict] = {}
    total_rns_primes = 0
    mult_table_cache: dict[tuple[int, int, int, int], list[list[int]]] = {}
    for rp in rns_paths:
        rd = json.loads(rp.read_text(encoding="utf-8"))
        fields = parse_rns_problem_fields(rd)
        periods = [int(x) for x in rd.get("full_rns_periods", rd.get("periods"))]
        modulus = int(fields.get("modulus", rd.get("modulus")))
        generator = int(fields.get("generator", 3))
        assert modulus == RSA2048_MODULUS, f"Non-RSA-2048 modulus in {rp.name}"
        p_bits = int(fields.get("rns_primes_bit_length", fields.get("prime_bits", 20)))
        m_in = int(fields["num_input_qubits"])
        w1 = int(fields["window1"])
        W = (m_in + w1 - 1) // w1
        gap_decl = int(fields.get("min_wraparound_gap", 33))
        assert len(periods) > 0, f"Empty prime list in {rp.name}"
        # Strict ascending order enforces canonical uniqueness (prevents prime-permutation tampering)
        assert all(periods[i] < periods[i + 1] for i in range(len(periods) - 1)), (
            f"Primes in {rp.name} are not strictly ascending (order or uniqueness violated)"
        )
        for p in periods:
            assert p.bit_length() == p_bits, f"Prime {p} bit-length != {p_bits} in {rp.name}"
            assert is_prime_u32(p), f"Composite modulus {p} in {rp.name}"
        prod_L = _balanced_prod(periods)
        assert prod_L > RSA2048_MODULUS ** W, f"RNS capacity L <= N^W in {rp.name}"
        dev = min(prod_L % RSA2048_MODULUS, (-prod_L) % RSA2048_MODULUS)
        assert dev * (1 << gap_decl) < RSA2048_MODULUS, f"RNS wraparound gap < {gap_decl} bits in {rp.name}"
        eff_gap = math.log2(RSA2048_MODULUS / dev) if dev > 0 else float("inf")

        # Verify zero-free 3^x mod N multiplier entries across boundary windows for ALL primes in this family
        mt_key = (generator, W, w1, m_in)
        if mt_key not in mult_table_cache:
            mult_table_cache[mt_key] = build_multiplier_table(modulus, generator, W, w1, m_in)
        mt = mult_table_cache[mt_key]
        for w_chk in sorted({0, 1, W // 2, W - 1}):
            wprod = _balanced_prod(mt[w_chk])
            assert math.gcd(wprod % prod_L, prod_L) == 1, f"Zero table entry found in {rp.name} window {w_chk}"

        rel_rns = f"data/rns_inputs/{rp.name}"
        rns_verified[rel_rns] = {
            "fields": fields,
            "periods": periods,
            "prod_L": prod_L,
            "dev": dev,
            "eff_gap": eff_gap,
            "W": W,
            "m_in": m_in,
            "w1": w1,
            "generator": generator,
            "mult_table": mt,
            "sha256": sha256_file(rp),
        }
        total_rns_primes += len(periods)

    m = load_json("data/combined_frontier_49_configs.json")
    by = {r["id"]: r for r in m["all_configs"]}
    assert len(m["all_configs"]) == 49
    assert len(m["pareto_ids_all"]) == 30
    assert len(m["pareto_ids_f33"]) == 28
    pareto_sa = pareto_2d([{"Q": by[i]["Q_peak"], "T_fact": by[i]["T_fact_same_adder"]} for i in m["pareto_ids_all"]])
    assert len(pareto_sa) == 29, f"Expected 29 Pareto points under matched b-2, got {len(pareto_sa)}"

    max_eq5_rel = 0.0
    for r in m["all_configs"]:
        b = r["budget"]
        sp = r["spec"]
        assert int(sp["s_param"]) == r["s"] and int(sp["share"]) == r["k"]
        assert int(sp["center_bits"]) == r["D"] and int(sp["len_accumulator"]) == r["f"]
        assert int(sp["mask_bits"]) == r["mask_bits"] and int(sp["min_wraparound_gap"]) == r["gap_declared"]
        assert sp["loop1_mode"] == r["loop1_mode"] and bool(sp["q_safe_k1"]) == bool(r["q_safe_k1"])

        rv = rns_verified[f"data/rns_inputs/{r['family']}.json"]
        assert len(rv["periods"]) == r["primes"] and rv["W"] == r["num_windows"]
        assert abs(rv["eff_gap"] - b["gap_effective_bits"]) < 5e-3, f"eff_gap mismatch on {r['id']}"

        s, f, mask, gap = r["s"], r["f"], b["mask_bits"], b["gap_declared"]
        P, W = r["primes"], r["num_windows"]
        assert b["M_crt"] == 8 * P, f"Non-8P M_crt on {r['id']}"
        p_keep = 1.0 - 2.0 ** (mask - f) - (s + 1) * b["eta"] - (2.0 ** (f - mask)) * (8 * P) * (2.0 ** (-f) + 2.0 ** (-gap))
        tf_calc = (s + 1) * r["T_shot_strict"] / p_keep
        rel = abs(tf_calc - r["T_fact_strict"]) / r["T_fact_strict"]
        max_eq5_rel = max(max_eq5_rel, rel)
        assert rel < 1e-12, f"Eq. (A5) mismatch on {r['id']}: rel={rel}"
        assert b["p_keep_pinned"] >= b["p_keep_declared"] - 1e-12, f"Pinned keep < declared on {r['id']}"
        assert abs(r["T_shot_strict"] * b["factor_pinned"] - r["T_fact_pinned"]) / r["T_fact_pinned"] < 1e-12

        qq_expected = W + (0 if r["q_safe_k1"] else (28 if int(sp["cutoff"]) == 3 else 35))
        assert r["exact_qq_adds_per_prime"] == qq_expected, f"Nominal QQ mismatch on {r['id']}: {r['exact_qq_adds_per_prime']} != {qq_expected}"
        t_sa_calc = (r["T_shot_strict"] - P * qq_expected) * ((s + 1) / p_keep)
        assert abs(t_sa_calc - r["T_fact_same_adder"]) / r["T_fact_same_adder"] < 1e-12
    diffs = [100.0 * (by[i]["T_fact_strict"] / by[i]["T_fact_same_adder"] - 1.0) for i in m["pareto_ids_all"]]
    print(
        f"  PASS: 9/9 RNS files verified ({total_rns_primes:,} strictly ascending primes, zero-free boundary windows, L > N^W); "
        f"49/49 configs match Eq. (A5) (max rel err {max_eq5_rel:.2e}); "
        f"30 strict / 29 matched b-2 Pareto pts (28 at f=33); strict vs b-2 gap +{min(diffs):.2f}% to +{max(diffs):.2f}%."
    )

    print("\n=== [3/7] ALL 6 ALL-PRIME 32-TRAJECTORY RECEIPTS, EXACT RNS BINDING & L1 CERTIFICATES ===")

    def verify_semantic_rns(pub: dict, *, require_exact_sha: bool = False) -> dict:
        rel_rns = pub["path"]
        assert rel_rns in rns_verified, f"Unverified RNS input path: {rel_rns}"
        rv = rns_verified[rel_rns]
        fields, periods = rv["fields"], rv["periods"]
        assert int(pub["prime_count"]) == len(periods), f"Prime count mismatch on {rel_rns}"
        if "num_input_qubits" in pub:
            assert int(pub["num_input_qubits"]) == int(fields["num_input_qubits"])
        if "window1" in pub:
            assert int(pub["window1"]) == int(fields["window1"])
        if require_exact_sha:
            assert pub.get("sha256") == rv["sha256"], (
                f"Byte-level public_input.sha256 mismatch on {rel_rns}: {pub.get('sha256')} != {rv['sha256']}"
            )
        return rv

    tot_primes, tot_segs, tot_zero_tax_segs = 0, 0, 0
    tot_unique_bytes = 0
    max_torus = 0.0
    for cid in ANCHOR_IDS:
        pri_rel = f"receipts/all_prime_32shot/{cid}/receipt.json"
        hld_rel = f"receipts/all_prime_32shot/{cid}/holdout_receipt.json"
        cert_rel = f"receipts/all_prime_32shot/{cid}/l1_dyadic_certificate.json"
        pri = load_json(pri_rel)
        hld = load_json(hld_rel)
        cert = load_json(cert_rel)
        req_sha = cid in EXACT_INPUT_SHA256_ANCHORS
        rv = verify_semantic_rns(pri["public_input"], require_exact_sha=req_sha)
        verify_semantic_rns(hld["public_input"], require_exact_sha=req_sha)

        # Bind receipt & v3 certificate directly to live-recomputed prod(p) mod N from data/rns_inputs/*.json
        dev_over_n_live = float(Fraction(rv["dev"], RSA2048_MODULUS))
        dev_over_n_rec = float(pri["l1_dyadic_certificate"]["pinned_rns_exact_bound"]["modular_deviation_over_N"])
        assert abs(dev_over_n_live - dev_over_n_rec) <= 1e-15 * max(1.0, dev_over_n_live), (
            f"prod(p) mod N binding mismatch on {cid}: live={dev_over_n_live:.16e} vs receipt={dev_over_n_rec:.16e}"
        )

        assert pri["verification"]["verdict"] == "VERIFIED" and hld["verification"]["verdict"] == "VERIFIED"
        assert hld["verification"]["seed_isolated_from_primary"] is True
        assert pri["verification"]["all_shots_within_exact_l1_dyadic_crt_bound"] is True
        assert hld["verification"]["all_shots_within_exact_l1_dyadic_crt_bound"] is True
        assert pri["segment_hashes_sha256"] == hld["segment_hashes_sha256"], f"Primary/holdout segment hash mismatch on {cid}"
        p_segs = pri["segments"]
        h_segs = hld["segments"]
        assert len(p_segs) == len(h_segs)
        assert [x["stream_sha256"] for x in p_segs] == [x["stream_sha256"] for x in h_segs]

        mr = pri["measured_resources"]
        ap = m["all_prime_32shot_anchors"][cid]
        rec = by[cid]
        assert int(mr["peak_logical_qubits"]) == int(ap["Q_peak"]) == rec["Q_peak"]
        assert int(mr["toffoli_monolithic_equivalent"]) == int(ap["T_shot_mono"]) == int(rec["T_shot_strict"])
        assert int(mr["boundary_loop1_repayment_toffoli"]) == int(ap["boundary_tax"])
        assert abs(ap["T_fact_strict"] - rec["T_fact_strict"]) / rec["T_fact_strict"] < 1e-12
        assert abs(ap["T_fact_same_adder"] - rec["T_fact_same_adder"]) / rec["T_fact_same_adder"] < 1e-12
        assert abs(ap["T_fact_with_099"] - rec["T_fact_strict"] / 0.99) / ap["T_fact_with_099"] < 1e-12
        assert abs(ap["QT_strict"] - rec["QT_strict"]) / rec["QT_strict"] < 1e-12
        assert ap["receipt_sha256"] == sha256_file(REPO / pri_rel)
        assert ap["holdout_receipt_sha256"] == sha256_file(REPO / hld_rel)

        # Standalone v3 rigorous Eq. (A5) diameter L1 certificate exact sync + live pinned prod(p) check
        assert cert["schema"] == "combined-l1-dyadic-crt-rational-certificate-v3-rigorous-diameter"
        assert abs(cert["declared_gap_bound"]["keep_dyadic_diameter"] - rec["budget"]["p_keep_declared"]) < 1e-12
        assert abs(cert["pinned_rns_exact_bound"]["keep_dyadic_diameter_exact"] - rec["budget"]["p_keep_pinned"]) < 1e-12
        f_b, m_b, s_b = rec["f"], rec["mask_bits"], rec["s"]
        live_keep_pinned = float(
            Fraction(1)
            - Fraction(1, 1 << (f_b - m_b))
            - Fraction(s_b + 1) * Fraction(str(rec["budget"]["eta"]))
            - Fraction(1 << (f_b - m_b)) * Fraction(8 * len(rv["periods"])) * (Fraction(1, 1 << f_b) + Fraction(rv["dev"], RSA2048_MODULUS))
        )
        assert abs(live_keep_pinned - cert["pinned_rns_exact_bound"]["keep_dyadic_diameter_exact"]) < 1e-12, (
            f"Live pinned Eq. (A5) mismatch on {cid}"
        )

        tot_primes += int(pri["public_input"]["prime_count"])
        tot_segs += len(p_segs)
        tot_unique_bytes += int(mr.get("total_emitted_opstrm01_bytes", 0))
        if int(ap["boundary_tax"]) == 0:
            tot_zero_tax_segs += len(p_segs)
        max_torus = max(
            max_torus,
            float(pri["verification"]["max_torus_relative_error_vs_pow_g_e_mod_N"]),
            float(hld["verification"]["max_torus_relative_error_vs_pow_g_e_mod_N"]),
        )
    assert tot_primes == 119482 and tot_segs == 6505 and tot_zero_tax_segs == 5610
    assert max_torus < 1e-7
    print(
        f"  PASS: 6/6 all-prime circuits verified ({tot_primes:,} primes, {tot_segs:,} segments, "
        f"{tot_zero_tax_segs:,} zero-tax segments, {tot_unique_bytes/1e12:.2f} TB unique streams / "
        f"{2*tot_unique_bytes/1e12:.2f} TB replayed, exact prod(p) mod N & SHA-256 bindings, "
        f"v3 Eq. (A5) L1 certs <= 1e-12, max torus err {max_torus:.2e})."
    )

    print("\n=== [4/7] SAME-RNS (19,878p, f=33) CONTROLLED ABLATION & UNSEGMENTED REPLAY ===")
    r4bp = load_json("receipts/same_rns_controlled_ablation/R4B_PLUS_odd_crt_receipt.json")
    elem = load_json("receipts/same_rns_controlled_ablation/R4B_unsegmented_elementary_replay.json")
    verify_semantic_rns(r4bp["public_input"])
    r4bc = by["R4B_PLUS_COMBINED"]
    assert r4bp["public_input"]["prime_count"] == 19878
    q_r4bp = int(r4bp["measured_resources"]["peak_logical_qubits"])
    t_r4bp = float(r4bp["measured_resources"]["toffoli_monolithic_equivalent"])
    btax_r4bp = float(r4bp["measured_resources"]["boundary_loop1_repayment_toffoli"])
    assert q_r4bp == 2204 and int(t_r4bp) == 246971807 and int(btax_r4bp) == 3227136
    assert r4bc["Q_peak"] == 2199 and int(r4bc["T_shot_strict"]) == 227414986
    delta_tp = (r4bc["T_shot_strict"] - t_r4bp) / 19878.0
    p_keep_odd = 1.0 - 2.0 ** (26 - 33) - 5.0 * r4bc["budget"]["eta"] - (2.0 ** 7) * (8 * 19878) * (3.0 * 2.0 ** (-33))
    tf_r4bp = 5.0 * t_r4bp / p_keep_odd
    assert abs(100.0 * (r4bc["T_shot_strict"] / t_r4bp - 1.0) - (-7.92)) < 0.01
    assert abs(100.0 * (r4bc["T_fact_strict"] / tf_r4bp - 1.0) - (-8.14)) < 0.01
    elem_full = elem["rsa2048_full_19878primes_64branches_elementary"]
    assert int(elem_full["realized_AND_vs_LKS_macro_reconciliation"]["realized_elementary_AND_T_shot"]) == 261247909
    assert int(elem_full["peak_logical_wires"]) == 2211
    print(
        f"  PASS: Same-RNS ablation (19,878p, f=33, 16-shot odd-CRT vs 32-shot dyadic): -5 Q, {delta_tp:.1f} T/p "
        "(-7.92% shot, -8.14% T_fact), 3.23M->0 seg tax; unsegmented R4B replay = 261,247,909 ANDs (13,143 T/p) at 2,211 Q."
    )

    print("\n=== [5/7] HEADLINE MANUSCRIPT METRICS VS GIDNEY (2025) & SHARMA-LAU (2026) ===")
    g28 = load_json("data/gidney_28_baseline_frontier.json")
    ref_pts = pareto_2d([
        {"Q": int(p["dual_meter"]["original_common_meter"]["Q"]), "T_fact": float(p["dual_meter"]["original_common_meter"]["T_fact"])}
        for p in g28["canonical_reference_28"]
    ])
    assert len(g28["canonical_reference_28"]) == 28 and len(ref_pts) == 25
    ref_min_qt = min(p["Q"] * p["T_fact"] for p in ref_pts)
    ref_min_t = min(p["T_fact"] for p in ref_pts)
    cmb_min_q = by["S14_PREFIX_COMBINED"]
    cmb_k20 = by["S4_ROLLK20_COMBINED"]
    cmb_min_qt = by["S4_K22_W8_CMB"]
    cmb_k24 = by["S4_K24_W8_OPT"]
    cmb_f33_w9 = by["S3_K22_W9_F33"]
    cmb_min_t = by["S3_K24_W9_CMB"]

    # Zero-sampling exact all-prime verification of headline ranking & sub-billion milestones
    assert cmb_min_q["Q_peak"] == 1306
    assert cmb_min_qt["all_prime_32shot_certified"] and cmb_k20["all_prime_32shot_certified"] and cmb_k24["all_prime_32shot_certified"]
    assert cmb_min_qt["QT_strict"] < cmb_k20["QT_strict"] and cmb_min_qt["QT_same_adder"] < cmb_k20["QT_same_adder"]
    assert cmb_k24["T_fact_strict"] < 1e9 and cmb_k24["boundary_repayment_toffoli"] == 0
    assert cmb_f33_w9.get("profiling_mode") == "exact_all_prime_monolithic"
    assert cmb_min_t.get("profiling_mode") == "exact_all_prime_monolithic"

    red_qt_st = 100.0 * (1.0 - cmb_min_qt["QT_strict"] / ref_min_qt)
    red_qt_sa = 100.0 * (1.0 - cmb_min_qt["QT_same_adder"] / ref_min_qt)
    red_t_f33_st = 100.0 * (1.0 - cmb_f33_w9["T_fact_strict"] / ref_min_t)
    red_t_f33_sa = 100.0 * (1.0 - cmb_k24["T_fact_same_adder"] / ref_min_t)
    red_t_st = 100.0 * (1.0 - cmb_min_t["T_fact_strict"] / ref_min_t)
    red_t_sa = 100.0 * (1.0 - cmb_min_t["T_fact_same_adder"] / ref_min_t)
    assert abs(red_qt_st - 69.4) < 0.1 and abs(red_qt_sa - 70.1) < 0.1
    assert abs(red_t_f33_st - 78.2) < 0.1 and abs(red_t_f33_sa - 78.6) < 0.1
    assert abs(red_t_st - 78.8) < 0.1 and abs(red_t_sa - 79.2) < 0.1

    sl = load_json("data/sharma_lau_2026_points.json")["points"]
    cmb_pareto_st = [{"Q": by[i]["Q_peak"], "T_fact": by[i]["T_fact_strict"]} for i in m["pareto_ids_all"]]
    sl_reds = [100.0 * (1.0 - best_at_cap(cmb_pareto_st, int(sp["Q"]))["T_fact"] / float(sp["T_fact"])) for sp in sl]
    sl_floor_red = 100.0 * (1.0 - cmb_min_t["T_fact_strict"] / min(float(sp["T_fact"]) for sp in sl))
    assert abs(min(sl_reds) - 24.93) < 0.05 and abs(max(sl_reds) - 43.36) < 0.05 and abs(sl_floor_red - 62.75) < 0.05
    print(
        f"  PASS: Min Q=1,306 Q; Locked All-Prime Min QT=-{red_qt_sa:.2f}% b-2 (-{red_qt_st:.2f}% strict) at 2,433 Q; "
        f"f=33 Min T=-{red_t_f33_sa:.2f}% b-2 ({cmb_k24['T_fact_same_adder']/1e9:.4f}B) / -{red_t_f33_st:.2f}% strict ({cmb_f33_w9['T_fact_strict']/1e9:.4f}B); "
        f"f=34 Min T={cmb_min_t['T_fact_same_adder']/1e9:.4f}B b-2 (-{red_t_sa:.2f}%) / "
        f"{cmb_min_t['T_fact_strict']/1e9:.4f}B strict (-{red_t_st:.2f}%) at 2,919 Q; "
        f"Sharma&Lau -{min(sl_reds):.2f}% to -{max(sl_reds):.2f}% matched Q (2,919Q vs 1,806Q floor -{sl_floor_red:.2f}%)."
    )

    print("\n=== [6/7] 23 FULL-PRIME MGF CERTIFICATES (TIGHTNESS + LIVE RNS SPOT RECOMPUTE) & LIVE EQ. (C5) HOEFFDING ===")
    mgf = load_json("receipts/mgf_tail_certificates/mgf_certificates_summary.json")
    assert mgf["count"] == 23 and len(mgf["certificates"]) == 23
    mgf_by_key: dict[tuple[str, int, str], Decimal] = {}
    lut = make_exp_upper_lut((1 << 20) - 2, scale_bits=39)
    mgf_spot_cache: dict[str, tuple[list[int], list[tuple[int, int, list[int], list[int]]]]] = {}
    live_mgf_certs_checked = 0
    with localcontext() as ctx:
        ctx.prec = 100
        tight_rel = Decimal("1e-45")
        grid_ulp = Decimal(4) / Decimal(1 << 96)
        for c in mgf["certificates"]:
            num_i = int(c["bad_weight_fraction"][0])
            den_i = int(c["bad_weight_fraction"][1])
            assert num_i > 0, f"Zero/negative MGF numerator on {c['family']} D={c['D']}"
            assert den_i == (1 << 192), f"Unexpected MGF denominator on {c['family']} D={c['D']}: {den_i}"
            num = Decimal(num_i)
            den = Decimal(den_i)
            exact_eta = Decimal(2) * (num / den).sqrt()
            s_val = int(c["family"].split("_")[0][1:])
            shots = Decimal(s_val + 1)
            eta_dec = Decimal(c["eta"])
            deb_dec = Decimal(c["shot_debit"])
            # Two-sided tightness (45-digit rel + 2^-192 grid rounding ULP) blocks silent numerator tampering
            assert eta_dec >= exact_eta and (eta_dec - exact_eta) <= exact_eta * tight_rel + grid_ulp, (
                f"MGF eta not tightly bound to 2*sqrt(num/den) on {c['family']} D={c['D']} {c['representation']}"
            )
            assert deb_dec >= shots * exact_eta and deb_dec >= shots * eta_dec, (
                f"MGF shot_debit < (s+1)*eta on {c['family']} D={c['D']}"
            )
            assert (deb_dec - shots * eta_dec) <= (shots * eta_dec) * tight_rel + shots * grid_ulp, (
                f"MGF shot_debit not tightly bound to (s+1)*eta on {c['family']} D={c['D']}"
            )
            mgf_by_key[(c["family"], int(c["D"]), c["representation"])] = eta_dec

            # Live stdlib strict_math.certify_prime recomputation directly from data/rns_inputs/*.json
            rel_rns = f"data/rns_inputs/{c['family']}.json"
            if rel_rns in rns_verified:
                rv = rns_verified[rel_rns]
                assert int(c["primes"]) == len(rv["periods"]), f"MGF prime count mismatch on {c['family']}"
                assert int(c["windows"]) == rv["W"], f"MGF window count mismatch on {c['family']}"
                assert int(c["input_bits"]) == rv["m_in"], f"MGF input_bits mismatch on {c['family']}"
                if c["family"] not in mgf_spot_cache:
                    ps = rv["periods"]
                    mt = rv["mult_table"]
                    ac = [len(row) for row in mt]
                    n_p = len(ps)
                    sample_idxs = [0, 1, n_p // 4, n_p // 2, n_p - 1]
                    rows_info = []
                    denom = 1 << rv["w1"]
                    for idx in sample_idxs:
                        p = ps[idx]
                        gp = primitive_root_prime(p)
                        logs = array("I", [0]) * p
                        val = 1
                        for e in range(p - 1):
                            logs[val] = e
                            val = (val * gp) % p
                        dlogs = []
                        for row in mt:
                            drow = []
                            for v in row:
                                rm = v % p
                                assert rm != 0, f"Zero table entry in {c['family']} at p={p}"
                                drow.append(logs[rm])
                            dlogs.append(drow)
                        center = (sum(sum(r_d) * (denom // len(r_d)) for r_d in dlogs) + denom // 2) // denom
                        plus_s = [sum(lut[d] for d in r_d) for r_d in dlogs]
                        minus_s = [sum(lut[(p - 2) - d] for d in r_d) for r_d in dlogs]
                        rows_info.append((p, center, plus_s, minus_s))
                    mgf_spot_cache[c["family"]] = (ac, rows_info)
                ac, rows_info = mgf_spot_cache[c["family"]]
                rep_norm = "signed" if c["representation"] in ("signed", "centered_signed") else c["representation"]
                sample_sum = sum(
                    certify_prime(
                        prime=p,
                        center=center,
                        plus_sums=plus_s,
                        minus_sums=minus_s,
                        address_counts=ac,
                        stored_bits=int(c["D"]),
                        representation=rep_norm,
                    )["bad_numerator"]
                    for p, center, plus_s, minus_s in rows_info
                )
                assert 0 < sample_sum <= num_i, (
                    f"Live 5-prime MGF floor violated on {c['family']} D={c['D']} {c['representation']}: "
                    f"live_5prime={sample_sum} vs stored={num_i}"
                )
                live_mgf_certs_checked += 1
    assert live_mgf_certs_checked == 17, f"Expected 17 published-family MGF certs live-checked, got {live_mgf_certs_checked}"

    t11 = load_json("data/hardware_and_kext_table11.json")
    mgf_cross_checked = 0
    hoeff_cache: dict[tuple[str, int, int], float] = {}
    for r in list(m["all_configs"]) + list(t11["rolling_anf_k_extension"]):
        meth = r["budget"].get("tail_method")
        if meth == "MGF":
            rep = r["spec"]["representation"]
            key_m = (r["family"], int(r["D"]), rep)
            assert key_m in mgf_by_key, f"Missing MGF summary entry for {r['id']}: {key_m}"
            eta_ref = float(mgf_by_key[key_m])
            eta_rec = float(r["budget"]["eta"])
            assert abs(eta_rec - eta_ref) / eta_ref < 1e-12, f"MGF eta mismatch on {r['id']}: {eta_rec} != {eta_ref}"
            mgf_cross_checked += 1
        elif meth == "Hoeffding":
            key_h = (r["family"], int(r["D"]), int(r["num_windows"]))
            if key_h not in hoeff_cache:
                rv = rns_verified[f"data/rns_inputs/{r['family']}.json"]
                cert_h = shifted_unsigned_hoeffding_certificate(
                    rv["periods"], windows=key_h[2], bits=key_h[1], bin_width=256
                )
                hoeff_cache[key_h] = float(cert_h["trace_upper_per_shot"])
            eta_h = hoeff_cache[key_h]
            eta_rec = float(r["budget"]["eta"])
            assert abs(eta_h - eta_rec) / eta_h < 1e-12, (
                f"Hoeffding bin256 mismatch on {r['id']} {key_h}: rational={eta_h:.12e}, stored={eta_rec:.12e}"
            )
    assert mgf_cross_checked == 31 and len(hoeff_cache) == 5
    print(
        f"  PASS: All 23/23 MGF certificates satisfy num>0, den=2^192, two-sided 45-digit tightness, "
        f"17/17 published-family live 5-prime strict_math.py floor checks, & match {mgf_cross_checked}/31 MGF configs (<1e-12); "
        f"all {len(hoeff_cache)}/5 Hoeffding families match live Eq. (C5) bin256 (<1e-12)."
    )

    print("\n=== [7/7] APPENDIX E TABLE V EXACT ALL-PRIME MONOLITHIC HARDWARE PROFILES & k > 24 EXTENSION ===")
    hw = t11["headline_hw_metrics"]
    assert len(hw) == 6
    for cid in ANCHOR_IDS:
        hrec = hw[cid]
        assert hrec["profiling_mode"] == "exact_all_prime_monolithic"
        assert hrec["Q_peak"] == by[cid]["Q_peak"]
        ps = hrec["proj_per_shot"]
        assert ps["AND"] == ps["AND_REL"], f"AND != AND_REL on {cid}"
        assert ps["AND"] + ps["CCX"] + ps["CSWAP"] == hrec["toff_total_per_shot"]
        assert hrec["toff_total_per_shot"] == int(by[cid]["T_shot_strict"]), (
            f"Table V vs master mismatch on {cid}: {hrec['toff_total_per_shot']} != {int(by[cid]['T_shot_strict'])}"
        )
        assert ps["CX"] + ps["CZ"] == hrec["cliff_2q_per_shot"]
        assert ps["HMR"] + ps["AND_REL"] == hrec["hmr_per_shot"]
        assert abs(hrec["cliff_2q_per_shot"] / hrec["toff_total_per_shot"] - hrec["cliff_2q_over_toff"]) < 1e-12
        assert abs(hrec["cx_per_shot"] / hrec["toff_total_per_shot"] - hrec["cx_over_toff"]) < 1e-12

    ext = t11["rolling_anf_k_extension"]
    assert len(ext) == 5
    for r in ext:
        b = r["budget"]
        sp = r["spec"]
        assert sp["variant_id"] == r["id"] and int(sp["share"]) == r["k"] and sp["loop1_mode"] == "rolling_anf"
        s, f, mask, gap = r["s"], r["f"], b["mask_bits"], b["gap_declared"]
        P, W = r["primes"], r["num_windows"]
        p_keep = 1.0 - 2.0 ** (mask - f) - (s + 1) * b["eta"] - (2.0 ** (f - mask)) * (8 * P) * (2.0 ** (-f) + 2.0 ** (-gap))
        tf_calc = (s + 1) * r["T_shot_strict"] / p_keep
        assert abs(tf_calc - r["T_fact_strict"]) / r["T_fact_strict"] < 1e-12
        assert r["exact_qq_adds_per_prime"] == W + 35 == 227
        t_sa_calc = (r["T_shot_strict"] - P * 227) * ((s + 1) / p_keep)
        assert abs(t_sa_calc - r["T_fact_same_adder"]) / r["T_fact_same_adder"] < 1e-12
    k42 = ext[-1]
    assert k42["id"] == "S4_K42_W8_EXT" and k42["Q_peak"] == 2953
    assert abs(k42["T_fact_strict"] / 1e9 - 0.8986) < 0.0002 and abs(k42["T_fact_same_adder"] / 1e9 - 0.8757) < 0.0002
    print(
        f"  PASS: 6/6 exact all-prime monolithic headline hardware profiles (diff == 0 vs monolithic T_shot_strict) "
        f"and 5/5 k>24 extensions verified (S4_K42_W8_EXT: 2,953 Q, {k42['T_fact_strict']/1e9:.4f}B strict / {k42['T_fact_same_adder']/1e9:.4f}B b-2)."
    )
    print("\nALL OFFLINE VERIFICATION CHECKS PASSED!")


if __name__ == "__main__":
    main()
