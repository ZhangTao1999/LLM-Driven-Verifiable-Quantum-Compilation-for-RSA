#!/usr/bin/env python3
"""Live Compiler, Primitive Exhaustion, Fault-Injection, EGKMV1, and RSA-2048 Dual-Backend Verification (~30-90 s).

Executes our actual flat 1-bit gate-stream compiler (`src/compiler/`), clean-room `OPSTRM01`
symbolic all-2^M + randomized kickmix backends (`src/flat_stream/`), independent `EGKMV1`
elementary verifier (`src/elementary_verifier/`), and exact MGF primitives (`src/mgf_certificates/`)
to verify:
  1. Exhaustive small-width non-restoring ctz(p-1)-sliced Stage-(ii) division across ALL 2^D states
     (D in {8, 9, 10, 11}, t in {1, 2, 3, 4}, including rare wrapped branches U >= 2H).
  2. Exhaustive universal all-to-all CZ phase synthesis across ALL 2^n inputs and symbolic outcomes
     at n in {5, 6, 7, 8} costing EXACTLY 4, 7, 11, 17 Toffolis.
  3. Negative-control fault-injection suite across BOTH strict=True and strict=False modes:
     omitted constant-term global phase flip after MX, dirty QFREE recycle on |1>, corrupted
     AND uncomputation, CX(q,q)/CZ(q,q) wire aliasing, dead-wire access, and double free.
  4. Live measurement of Stage-(ii) non-restoring division across ALL 19,878 real S4_P20_W8 RSA-2048
     primes confirming 237.96 (238.0) Toffolis/prime.
  5. Live measurement of Stage-(iv) dyadic torus mod-2^33 CRT confirming EXACT 442.00 Toffolis/prime
     under b0=10 (2-bit terminal selector) and EXACT 653.00 Toffolis/prime under b0=9 (3-bit selector),
     plus nominal Stage-(i) + exact Stage-(iii)/(iv) quantum-quantum adder counts (W + 35 / W + 28 [non-Pareto cutoff=3] / W).
  6. Live full-width 2048-bit RSA-2048 1-batch dual-backend verification (symbolic all-2^M phase check
     + serialized OPSTRM01 streaming randomized kickmix) of S14_PREFIX_COMBINED at 1,306 logical qubits.
  7. Independent EGKMV1 elementary verifier smoke & fault-injection test, plus live
     mgf_certificates.strict_math shifted_unsigned Chernoff check and rolling-ANF Stage-(i) check.
"""
from __future__ import annotations

import io
from pathlib import Path
import random
import struct
import sys
import tempfile

if sys.version_info < (3, 10):
    raise SystemExit("Python >= 3.10 is required to run this verification script.")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from flat_stream import Builder, Simulator, OP_QALLOC, OP_QFREE, OP_X, OP_CX, OP_Z, OP_CZ, OP_OXOR
from flat_stream.evalcore import StandaloneKickmixEngine, TestCase, _CompactVM, run_task
import compiler.primitives as cp
import compiler.bridge as cfb
from elementary_verifier.elementary_kickmix_verifier_ref import (
    GateViolationError,
    MAGIC as EGKM_MAGIC,
    SplitMix64,
    verify_egkm_stream,
)
from mgf_certificates.strict_math import (
    certify_prime_shifted_unsigned,
    make_exp_upper_lut,
    verify_exp_upper_lut,
)


def test_1_exhaustive_stage2_division() -> None:
    print("[1/7] Exhaustive Stage-(ii) non-restoring ctz(p-1) division (D in {8, 9, 10, 11}, all 2^D states incl. wrapped)...")
    cases = [
        (8, 5, 22),   # p=23, p-1=22=2^1*11 (t=1)
        (8, 5, 28),   # p=29, p-1=28=2^2*7  (t=2)
        (8, 5, 24),   # 2^3*3 (t=3)
        (9, 6, 46),   # p=47, p-1=46=2^1*23 (t=1)
        (9, 6, 52),   # p=53, p-1=52=2^2*13 (t=2)
        (9, 6, 40),   # p=41, p-1=40=2^3*5  (t=3)
        (9, 6, 48),   # 2^4*3 (t=4)
        (10, 6, 46),  # p=47, D-ell=4 (1,024 states)
        (10, 6, 52),  # p=53, D-ell=4 (t=2)
        (11, 7, 100), # p=101, p-1=100=2^2*25 (2,048 states)
        (11, 7, 112), # p=113, p-1=112=2^4*7  (t=4)
    ]
    for D, ell, mod in cases:
        n_states = 1 << D
        all_u = list(range(n_states))
        H2 = (1 << (D - ell)) * mod

        b_fwd = Builder()
        w = b_fwd.qubits(D)
        b_fwd.qubit_register(w)
        cp.emit_fast_nonrestoring_loop2(b_fwd, w, mod, ell, inverse=False)
        sim_fwd = Simulator(b_fwd, batch=n_states, strict=True)
        sim_fwd.write(w, all_u)
        sim_fwd.run()
        sim_fwd.assert_clean(w, phase=0)
        outs_fwd = sim_fwd.read(w)
        mask_r = (1 << ell) - 1
        for u in range(H2):
            assert (outs_fwd[u] & mask_r) == (u % mod), f"Wrong remainder D={D}, mod={mod}, u={u}"

        b_rt = Builder()
        w2 = b_rt.qubits(D)
        b_rt.qubit_register(w2)
        cp.emit_fast_nonrestoring_loop2(b_rt, w2, mod, ell, inverse=False)
        cp.emit_fast_nonrestoring_loop2(b_rt, w2, mod, ell, inverse=True)
        sim_rt = Simulator(b_rt, batch=n_states, strict=True)
        sim_rt.write(w2, all_u)
        sim_rt.run()
        sim_rt.assert_clean(w2, phase=0)
        assert sim_rt.read(w2) == all_u, f"Bijection failed on wrapped states D={D}, mod={mod}"
    print("  PASS: Exact remainder on all safe states U < 2H & exact bijection across all 2^D states (incl. wrapped U >= 2H)!")


def test_2_universal_cz_phase_synthesis() -> None:
    print("[2/7] Exhaustive universal all-to-all CZ phase synthesis at n in {5, 6, 7, 8}...")
    rng = random.Random(20261120)
    expected_k = {5: 4, 6: 7, 7: 11, 8: 17}
    n_coins = 3
    for n, k_want in expected_k.items():
        n_states = 1 << n
        truths = [[rng.randint(0, 1) for _ in range(n_states)] for _ in range(n_coins)]
        tt_words = np.array(
            [[sum(truths[c][x] << c for c in range(n_coins))] for x in range(n_states)],
            dtype=np.uint64,
        )

        b = Builder()
        addr = b.qubits(n)
        b.qubit_register(addr)
        anc = b.qubit()
        out_ms = [b.hmr(anc) for _ in range(n_coins)]
        b.recycle(anc)
        t0 = b.phase_summary()["total_emitted_toffoli"]
        cp.emit_universal_cz_phaseup(b, addr, tt_words, out_ms)
        toff = b.phase_summary()["total_emitted_toffoli"] - t0
        assert toff == k_want, f"Universal CZ n={n}: got {toff} Toffolis, expected {k_want}"

        sim = Simulator(b, batch=n_states, strict=True)
        sim.write(addr, list(range(n_states)))
        sim.run()
        assert sim.read(addr) == list(range(n_states))
        out_set = set(addr)
        assert not [w for w in sim._live if w not in out_set]
        for x in range(n_states):
            want_p = {c + 1 for c in range(n_coins) if truths[c][x]}
            assert sim._phase_poly[x] == want_p, f"Phase mismatch at n={n}, x={x}: {sim._phase_poly[x]} != {want_p}"
    print("  PASS: Exact 4, 7, 11, 17 Toffolis at n=5, 6, 7, 8 across all 2^n inputs & all 2^3 symbolic outcomes!")


def _assert_raises(fn, exc_types, label: str) -> None:
    try:
        fn()
    except exc_types:
        return
    raise AssertionError(f"Fault injection was NOT caught: {label}")


def test_3_negative_control_fault_injection() -> None:
    print("[3/7] Fault-injection suite in BOTH strict=True (symbolic) and strict=False (kickmix) modes...")

    # (a) Omitted constant-term global phase flip after MX
    b_ph = Builder()
    x = b_ph.qubits(2)
    b_ph.qubit_register(x)
    anc = b_ph.qubit()
    b_ph.x(anc)
    out_m = b_ph.hmr(anc)
    b_ph.recycle(anc)
    for strict in (True, False):
        # For strict=False, pick a seed whose SHAKE-256 coin bit for k=0 (m_id=1) is 1
        found_bad = False
        for sd in ([0] if strict else range(1, 32)):
            sim_bad = Simulator(b_ph, batch=4, strict=strict, seed=sd)
            sim_bad.write(x, [0, 1, 2, 3])
            sim_bad.run()
            try:
                sim_bad.assert_clean(x, phase=0)
            except AssertionError:
                found_bad = True
                break
        assert found_bad, f"Omitted constant-term global phase flip not caught (strict={strict})"

    b_ph.neg(out_m)
    for strict in (True, False):
        sim_ok = Simulator(b_ph, batch=4, strict=strict, seed=20261120)
        sim_ok.write(x, [0, 1, 2, 3])
        sim_ok.run()
        sim_ok.assert_clean(x, phase=0)

    # (b) Dirty QFREE recycle on |1> wire in BOTH strict=True and strict=False
    b_dirty = Builder()
    q0 = b_dirty.qubit()
    b_dirty.x(q0)
    b_dirty.recycle(q0)
    for strict in (True, False):
        sim = Simulator(b_dirty, batch=2, strict=strict, seed=7)
        _assert_raises(sim.run, AssertionError, f"dirty QFREE (strict={strict})")

    # (c) Corrupted AND release (OP_AND_REL when target != a & b) caught via explicit HMR+CZ phase kickback in BOTH modes
    b_and = Builder()
    a, bb = b_and.qubit(), b_and.qubit()
    b_and.qubit_register([a, bb])
    t = b_and.and_compute(a, bb)
    b_and.x(t)  # corrupt target before release!
    b_and.and_release(t, a, bb)
    for strict in (True, False):
        caught_and = False
        for sd in ([0] if strict else range(1, 32)):
            sim = Simulator(b_and, batch=4, strict=strict, seed=sd)
            sim.write([a, bb], [0, 1, 2, 3])
            try:
                sim.run()
                sim.assert_clean([a, bb], phase=0)
            except AssertionError:
                caught_and = True
                break
        assert caught_and, f"corrupted AND_REL phase kickback not caught (strict={strict})"

    # (d) Wire aliasing CX(q, q), CZ(q, q), dead wire access, negative wire ID, undefined outcome ID,
    #     OP_OXOR undefined handles, write() bounds, and double free in BOTH modes
    for strict in (True, False):
        sim = Simulator(batch=2, strict=strict, seed=11)
        _assert_raises(lambda: sim._exec_ops([(OP_QALLOC, -1)]), AssertionError, f"negative QALLOC (strict={strict})")
        sim._exec_ops([(OP_QALLOC, 0), (OP_QALLOC, 1)])
        _assert_raises(lambda: sim.write([0, 1], [-1, 0]), ValueError, f"negative write (strict={strict})")
        _assert_raises(lambda: sim.write([0, 1], [4, 0]), ValueError, f"oversized write (strict={strict})")
        _assert_raises(lambda: sim._exec_ops([(OP_CX, 0, 0)]), AssertionError, f"CX(q,q) alias (strict={strict})")
        _assert_raises(lambda: sim._exec_ops([(OP_CZ, 0, 0, -1)]), AssertionError, f"CZ(q,q) alias (strict={strict})")
        _assert_raises(lambda: sim._exec_ops([(OP_Z, 0, 9999)]), AssertionError, f"undefined outcome Z (strict={strict})")
        _assert_raises(lambda: sim._exec_ops([(OP_CZ, 0, 1, 9999)]), AssertionError, f"undefined outcome CZ (strict={strict})")
        _assert_raises(lambda: sim._exec_ops([(OP_OXOR, 0, 9999)]), AssertionError, f"undefined outcome OXOR src (strict={strict})")
        _assert_raises(lambda: sim._exec_ops([(OP_X, 99)]), AssertionError, f"dead wire X (strict={strict})")
        sim._exec_ops([(OP_QFREE, 0), (OP_QFREE, 1)])
        _assert_raises(lambda: sim._exec_ops([(OP_QFREE, 0)]), AssertionError, f"double QFREE (strict={strict})")

    # (d2) Independent StandaloneKickmixEngine fault-injection parity check (zero Simulator dependency)
    eng = StandaloneKickmixEngine(batch=2, seed=11)
    _assert_raises(lambda: eng._exec_ops([(OP_QALLOC, -1)]), AssertionError, "StandaloneKickmixEngine negative QALLOC")
    eng._exec_ops([(OP_QALLOC, 0), (OP_QALLOC, 1)])
    _assert_raises(lambda: eng.write([0, 1], [-1, 0]), ValueError, "StandaloneKickmixEngine negative write")
    _assert_raises(lambda: eng.write([0, 1], [4, 0]), ValueError, "StandaloneKickmixEngine oversized write")
    _assert_raises(lambda: eng._exec_ops([(OP_CX, 0, 0)]), AssertionError, "StandaloneKickmixEngine CX(q,q) alias")
    _assert_raises(lambda: eng._exec_ops([(OP_CZ, 0, 0, -1)]), AssertionError, "StandaloneKickmixEngine CZ(q,q) alias")
    _assert_raises(lambda: eng._exec_ops([(OP_Z, 0, 9999)]), AssertionError, "StandaloneKickmixEngine undefined outcome Z")
    _assert_raises(lambda: eng._exec_ops([(OP_OXOR, 0, 9999)]), AssertionError, "StandaloneKickmixEngine undefined outcome OXOR")
    _assert_raises(lambda: eng._exec_ops([(OP_X, 99)]), AssertionError, "StandaloneKickmixEngine dead wire X")
    eng._exec_ops([(OP_QFREE, 0), (OP_QFREE, 1)])
    _assert_raises(lambda: eng._exec_ops([(OP_QFREE, 0)]), AssertionError, "StandaloneKickmixEngine double QFREE")

    # (e) OPSTRM01 _CompactVM & run_task schema enforcement (empty stream, dropped register, wrong width) + framing rejections
    class _TwoRegTask:
        num_tests = 2
        seed_domain = b"schema-neg-test"
        register_widths = [4, 4]

        def gen_test_indexed(self, index: int, rng) -> TestCase:
            return TestCase(inputs=[index + 1, index + 2], expected=[index + 1, index + 2])

    empty_stream = b"OPSTRM01" + struct.pack("<II", 0, 0)
    code_e, rep_e = run_task(_TwoRegTask(), ".", io.BytesIO(empty_stream), b"\x00" * 32)
    assert code_e == 1 and rep_e["verdict"] == "FAIL", f"run_task accepted empty 0-reg stream: {rep_e}"

    dropped_reg_stream = b"OPSTRM01" + struct.pack("<III4I", 1, 0, 4, 0, 1, 2, 3)
    code_d, rep_d = run_task(_TwoRegTask(), ".", io.BytesIO(dropped_reg_stream), b"\x00" * 32)
    assert code_d == 1 and rep_d["verdict"] == "FAIL", f"run_task accepted dropped output register: {rep_d}"

    wrong_width_stream = b"OPSTRM01" + struct.pack("<III3II4I", 2, 0, 3, 0, 1, 2, 4, 3, 4, 5, 6)
    code_w, rep_w = run_task(_TwoRegTask(), ".", io.BytesIO(wrong_width_stream), b"\x00" * 32)
    assert code_w == 1 and rep_w["verdict"] == "FAIL", f"run_task accepted wrong register width: {rep_w}"

    class _ZeroSecondRegTask:
        num_tests = 2
        seed_domain = b"freed-output-neg-test"
        register_widths = [2, 2]

        def gen_test_indexed(self, index: int, rng) -> TestCase:
            return TestCase(inputs=[index + 1, 0], expected=[index + 1, 0])

    # Malicious stream freeing expected-zero output register wires 2, 3 via OP_QFREE
    freed_out_stream = (
        b"OPSTRM01"
        + struct.pack("<III2II2I", 2, 2, 2, 0, 1, 2, 2, 3)
        + struct.pack("<BBIBBI", OP_QFREE, 1, 2, OP_QFREE, 1, 3)
    )
    code_f, rep_f = run_task(_ZeroSecondRegTask(), ".", io.BytesIO(freed_out_stream), b"\x00" * 32)
    assert code_f == 1 and rep_f["verdict"] == "FAIL", f"run_task accepted freed output register: {rep_f}"

    valid_1reg_hdr = struct.pack("<III2I", 1, 0, 2, 0, 1)
    bad_trail = io.BytesIO(valid_1reg_hdr + b"\x99")
    _assert_raises(
        lambda: _CompactVM(bad_trail, [TestCase([0], [0])], Simulator(batch=1, strict=False, seed=1)).run(),
        ValueError,
        "OPSTRM01 trailing byte after declared n_ops",
    )
    bad_trunc = io.BytesIO(struct.pack("<III2IBB", 1, 1, 2, 0, 1, OP_QALLOC, 1) + b"\x01\x00")
    _assert_raises(
        lambda: _CompactVM(bad_trunc, [TestCase([0], [0])], Simulator(batch=1, strict=False, seed=1)).run(),
        ValueError,
        "OPSTRM01 truncated opcode payload",
    )
    bad_arity = io.BytesIO(struct.pack("<III2IBB2i", 1, 1, 2, 0, 1, OP_QALLOC, 2, 2, 3))
    _assert_raises(
        lambda: _CompactVM(bad_arity, [TestCase([0], [0])], Simulator(batch=1, strict=False, seed=1)).run(),
        ValueError,
        "OPSTRM01 invalid opcode arity",
    )

    print("  PASS: All 13 dual-mode fault-injection classes, 3 schema rejections, and 3 OPSTRM01 framing rejections caught immediately!")


def test_4_live_stage2_all_19878_primes() -> None:
    print("[4/7] Live Stage-(ii) non-restoring average across ALL 19,878 real RSA-2048 S4_P20_W8 primes...")
    prob = cfb.load_public_rns_problem(
        REPO / "data/rns_inputs/S4_P20_W8.json", max_primes=None, name="S4_P20_W8", spec=cfb.R4B_PLUS_COMBINED_SPEC
    )
    orig_emit = Builder._emit_op
    Builder._emit_op = lambda self, op: None
    try:
        t_tot = 0
        for p in prob.periods:
            b = Builder()
            lane = b.qubits(26)
            cp.emit_fast_nonrestoring_loop2(b, lane, p - 1, 20, inverse=False)
            cp.emit_fast_nonrestoring_loop2(b, lane, p - 1, 20, inverse=True)
            t_tot += b.phase_summary()["total_emitted_toffoli"]
    finally:
        Builder._emit_op = orig_emit
    avg = t_tot / len(prob.periods)
    assert abs(avg - 237.958) < 0.01, f"Unexpected Stage (ii) avg: {avg}"
    print(f"  PASS: Exact live Stage-(ii) fwd+rev across 19,878 primes = {avg:.3f} T/p ({t_tot:,} total).")


def test_5_live_stage4_crt_and_exact_qq_adders() -> None:
    print("[5/7] Live Stage-(iv) dyadic CRT (442 vs 653 vs 1,076 vs 1,332 T/p) & nominal QQ-adder count (W+35 vs W+28 [non-Pareto] vs W)...")
    orig_emit = Builder._emit_op
    Builder._emit_op = lambda self, op: None
    orig_qq = cfb.emit_qq_add
    orig_crt = cfb._emit_dyadic_crt_prime
    orig_pk = cfb._emit_run_prime_combined
    try:
        spec_b10 = cfb.CombinedScheduleSpec(
            variant_id="S4_K22_W8_CMB", s_param=4, share=22, center_bits=26,
            len_accumulator=33, mask_bits=26, min_wraparound_gap=33,
            b0=10, stage_widths=(3, 3, 2), terminal_sel_bits=2, cutoff=2,
            startup_lam=8, stage_lam=2, loop1_mode="rolling_anf", q_safe_k1=False,
        )
        spec_cut3 = cfb.CombinedScheduleSpec(
            variant_id="S14_K2_W6_CMB", s_param=14, share=2, center_bits=26,
            len_accumulator=33, mask_bits=26, min_wraparound_gap=33,
            b0=8, stage_widths=(3, 3, 2), terminal_sel_bits=4, cutoff=3,
            startup_lam=4, stage_lam=2, loop1_mode="rolling_anf", q_safe_k1=False,
        )
        for label, rns_rel, spec, want_crt, want_pk_qq in [
            ("S4_K22_W8_CMB (b0=10, term_sel=2)", "data/rns_inputs/S4_P20_W8.json", spec_b10, 442.0, 35.0),
            ("R4B_PLUS_COMBINED (b0=9, term_sel=3)", "data/rns_inputs/S4_P20_W8.json", cfb.R4B_PLUS_COMBINED_SPEC, 653.0, 35.0),
            ("S14_K2_W6_CMB (cutoff=3, non-Cuccaro)", "data/rns_inputs/S14_P20_W6.json", spec_cut3, 1076.0, 28.0),
            ("S14_PREFIX_COMBINED (q_safe Cuccaro)", "data/rns_inputs/S14_P20_W6.json", cfb.S14_PREFIX_COMBINED_SPEC, 1332.0, 0.0),
        ]:
            prob = cfb.load_public_rns_problem(REPO / rns_rel, max_primes=40, name=label, spec=spec)
            cnt = {"pk_qq": 0, "crt_t": 0, "in_pk": False}

            def w_qq(b, *a, **kw):
                if cnt["in_pk"]:
                    cnt["pk_qq"] += 1
                return orig_qq(b, *a, **kw)

            def w_crt(b, *a, **kw):
                t0 = b.phase_summary()["total_emitted_toffoli"]
                r = orig_crt(b, *a, **kw)
                cnt["crt_t"] += b.phase_summary()["total_emitted_toffoli"] - t0
                return r

            def w_pk(*a, **kw):
                cnt["in_pk"] = True
                try:
                    return orig_pk(*a, **kw)
                finally:
                    cnt["in_pk"] = False

            cfb.emit_qq_add = w_qq
            cfb._emit_dyadic_crt_prime = w_crt
            cfb._emit_run_prime_combined = w_pk
            cfb.build_combined_segment(prob, spec, 0, 1, include_final_clear=False)
            crt_avg = cnt["crt_t"] / spec.share
            pk_qq_avg = cnt["pk_qq"] / spec.share
            assert abs(crt_avg - want_crt) < 1e-6, f"{label}: CRT {crt_avg} != {want_crt}"
            assert abs(pk_qq_avg - want_pk_qq) < 1e-6, f"{label}: PK QQ {pk_qq_avg} != {want_pk_qq}"
            print(f"  PASS: {label}: Stage-(iv) CRT = {crt_avg:.1f} T/p, PK QQ adders = {pk_qq_avg:.1f}/prime.")
    finally:
        cfb.emit_qq_add = orig_qq
        cfb._emit_dyadic_crt_prime = orig_crt
        cfb._emit_run_prime_combined = orig_pk
        Builder._emit_op = orig_emit


def test_6_live_rsa2048_dual_backend() -> None:
    print("[6/7] Live full-width RSA-2048 dual-backend (symbolic all-2^M + OPSTRM01 kickmix) on S14_PREFIX_COMBINED (1,306 Q)...")
    spec = cfb.S14_PREFIX_COMBINED_SPEC
    prob = cfb.load_public_rns_problem(
        REPO / "data/rns_inputs/S14_P20_W6.json", max_primes=2, name="S14_LIVE_CHECK", spec=spec
    )
    b, meta = cfb.build_combined_segment(prob, spec, 0, 1, include_final_clear=True)
    assert meta["peak_qubits"] == 1306, f"Expected 1,306 Q, got {meta['peak_qubits']}"

    rng = random.Random(42)
    init_pairs = [(rng.randrange(1 << prob.num_input_qubits), rng.randrange(1 << prob.len_accumulator)) for _ in range(4)]
    seg_cases = cfb.propagate_combined_cases(prob, spec, [(0, 1, True)], init_pairs)[0]
    tests = [TestCase(inputs=list(inp), expected=list(exp)) for inp, exp in seg_cases]
    all_out_wires = [w for reg in b.registers for w in reg]

    # Backend 1: Exact symbolic all-2^M outcome phase Simulator
    sim_sym = Simulator(b, batch=len(tests), strict=True)
    for idx, reg in enumerate(b.registers):
        sim_sym.write(reg, [t.inputs[idx] for t in tests])
    sim_sym.run()
    for idx, reg in enumerate(b.registers):
        assert sim_sym.read(reg) == [t.expected[idx] for t in tests], f"Symbolic basis mismatch on reg {idx}"
    sim_sym.assert_clean(all_out_wires, phase=0)

    # Backend 2: Serialized OPSTRM01 binary stream + decoupled StandaloneKickmixEngine (zero Simulator dependency)
    with tempfile.TemporaryDirectory() as td:
        ops_path = Path(td) / "s14_batch0.ops"
        with ops_path.open("wb") as wf:
            b.emit(meta=False, dest=wf)
        with ops_path.open("rb") as fh:
            magic = fh.read(8)
            assert magic == b"OPSTRM01"
            km_eng = StandaloneKickmixEngine(batch=len(tests), seed=20261120)
            vm = _CompactVM(fh, tests, km_eng)
            vm.run()
            for idx, (_, wires) in enumerate(vm.registers):
                assert km_eng.read(wires) == [t.expected[idx] for t in tests], f"Kickmix basis mismatch on reg {idx}"
            km_eng.assert_clean([w for _, wires in vm.registers for w in wires], phase=0)
            assert km_eng._peak_live == 1306
            assert km_eng.toffoli == sim_sym.toffoli
    print(
        f"  PASS: Full-width RSA-2048 S14_PREFIX_COMBINED verified at Q_peak={meta['peak_qubits']} Q "
        "(0 leaked wires, 0 symbolic all-2^M phase, decoupled StandaloneKickmixEngine OPSTRM01 PASS)!"
    )


def _pack_egkm_stream(mx_seed: int, corrupt_phase: bool = False, dirty_and: bool = False) -> bytes:
    """Build a minimal 4-branch EGKMV1 binary stream testing AND + MX_FREE + CZ cleanup."""
    rng = SplitMix64(mx_seed)
    m_bit = rng.next_bit()
    buf = bytearray()
    # Header: magic, mx_seed, num_branches=4, max_wires=4, num_init_wires=2 (wires 0, 1)
    buf += struct.pack("<8sQIII", EGKM_MAGIC, mx_seed, 4, 4, 2)
    # Init wires: wire 0 = [0,1,0,1] (0b1010 = 10), wire 1 = [0,0,1,1] (0b1100 = 12), both readonly (flags=1)
    buf += struct.pack("<HHQ", 0, 1, 0b1010)
    buf += struct.pack("<HHQ", 1, 1, 0b1100)
    # 1 output group of length 2: [0, 1]
    buf += struct.pack("<IIHH", 1, 2, 0, 1)
    # OP_ALLOC(2)
    buf += struct.pack("<I", 1 | (2 << 4))
    if dirty_and:
        buf += struct.pack("<I", 3 | (2 << 4))  # OP_X(2) makes AND target dirty!
    # OP_AND(0, 1, 2) -> wire 2 = 0 & 1 = 0b1000
    buf += struct.pack("<II", 8 | (0 << 4) | (1 << 18), 2)
    # OP_MX_FREE(2, m_bit) -> kicks phase by val[2] = val[0]&val[1] if m_bit==1
    buf += struct.pack("<I", 9 | (2 << 4) | (m_bit << 18))
    # Cancel phase via OP_CZ(0, 1) if m_bit == 1 (unless corrupt_phase=True)
    if m_bit and not corrupt_phase:
        buf += struct.pack("<I", 6 | (0 << 4) | (1 << 18))
    # OP_EOF
    buf += struct.pack("<I", 0)
    return bytes(buf)


def test_7_egkmv1_mgf_and_rolling_anf() -> None:
    print("[7/7] Independent EGKMV1 elementary verifier, shifted_unsigned MGF Chernoff, and rolling-ANF smoke tests...")

    # (a) EGKMV1 clean pass + dirty AND & residual phase fault injection
    # Choose seed whose first SplitMix64 bit is 1 so CZ cancellation is actually exercised
    seed1 = next(s for s in range(1, 100) if SplitMix64(s).next_bit() == 1)
    res = verify_egkm_stream(io.BytesIO(_pack_egkm_stream(seed1)))
    assert res["status"] == "PASS" and res["peak_live_wires"] == 3 and res["gate_counts"]["AND"] == 1
    assert res["output_groups"] == [[0, 1, 2, 3]]
    _assert_raises(
        lambda: verify_egkm_stream(io.BytesIO(_pack_egkm_stream(seed1, corrupt_phase=True))),
        GateViolationError,
        "EGKMV1 residual phase",
    )
    _assert_raises(
        lambda: verify_egkm_stream(io.BytesIO(_pack_egkm_stream(seed1, dirty_and=True))),
        GateViolationError,
        "EGKMV1 dirty AND target",
    )

    # (b) Exact integer-only shifted_unsigned MGF Chernoff certification on a small toy prime
    p_toy = 37  # ell=6, p-1=36
    # Use a small-prime lambda_num/lambda_den so exp(lambda*36) is well inside uint64
    lam_num, lam_den = 1, 16
    lut = make_exp_upper_lut(p_toy - 2, scale_bits=39, lambda_num=lam_num, lambda_den=lam_den)
    verify_exp_upper_lut(lut, scale_bits=39, lambda_num=lam_num, lambda_den=lam_den)
    rng = random.Random(20261120)
    n_win, addr_cnt = 12, 8
    dlogs = [[rng.randrange(0, p_toy - 1) for _ in range(addr_cnt)] for _ in range(n_win)]
    plus_sums = [sum(int(lut[v]) for v in w) for w in dlogs]
    minus_sums = [sum(int(lut[(p_toy - 2) - v]) for v in w) for w in dlogs]
    center = round(sum(sum(w) / addr_cnt for w in dlogs))
    cert = certify_prime_shifted_unsigned(
        prime=p_toy,
        center=center,
        plus_sums=plus_sums,
        minus_sums=minus_sums,
        address_counts=[addr_cnt] * n_win,
        stored_bits=11,  # H = 2^(11-6-1)*36 = 576 >> max sum 12*35=420
        lambda_num=lam_num,
        lambda_den=lam_den,
    )
    assert cert["representation"] == "shifted_unsigned" and cert["upper_failure_threshold"] == center + 576
    assert cert["half_width"] == 576 and cert["bad_numerator"] < (1 << 192)

    # (c) Live rolling-ANF Stage-(i) dual-backend verification on a small R4B_PLUS_COMBINED sub-batch
    spec_r = cfb.R4B_PLUS_COMBINED_SPEC
    prob_r = cfb.load_public_rns_problem(
        REPO / "data/rns_inputs/S4_P20_W8.json", max_primes=spec_r.share, name="R4B_ROLL_TEST", spec=spec_r
    )
    br, metar = cfb.build_combined_segment(prob_r, spec_r, 0, 1, include_final_clear=True)
    assert metar["peak_qubits"] == 2199, f"Expected 2,199 Q on R4B_PLUS_COMBINED, got {metar['peak_qubits']}"
    init_r = [(rng.randrange(1 << prob_r.num_input_qubits), rng.randrange(1 << prob_r.len_accumulator)) for _ in range(2)]
    seg_r = cfb.propagate_combined_cases(prob_r, spec_r, [(0, 1, True)], init_r)[0]
    sim_r = Simulator(br, batch=len(seg_r), strict=True)
    for idx, reg in enumerate(br.registers):
        sim_r.write(reg, [inp[idx] for inp, _ in seg_r])
    sim_r.run()
    for idx, reg in enumerate(br.registers):
        assert sim_r.read(reg) == [exp[idx] for _, exp in seg_r], f"Rolling-ANF basis mismatch on reg {idx}"
    sim_r.assert_clean([w for reg in br.registers for w in reg], phase=0)
    assert len(br._free_o) > 0, "Expected NibbleOutcomeSynthesizer.retire() to recycle derived outcome IDs"

    # (d) Default trace_upper=None path of compute_l1_dyadic_certificate (Eq. (C5) Hoeffding -> Eq. (A5) diameter)
    spec_s3 = cfb.S3_K22_W9_COMBINED_SPEC
    prob_s3 = cfb.load_public_rns_problem(
        REPO / "data/rns_inputs/S3_P20_W9.json", max_primes=None, name="S3_P20_W9", spec=spec_s3
    )
    l1_default = cfb.compute_l1_dyadic_certificate(prob_s3, spec_s3, trace_upper=None)
    assert l1_default["schema"] == "combined-l1-dyadic-crt-rational-certificate-v3-rigorous-diameter"
    assert 0.989 < l1_default["declared_gap_bound"]["keep_dyadic_diameter"] < 0.991
    assert l1_default["pinned_rns_exact_bound"]["keep_dyadic_diameter_exact"] > l1_default["declared_gap_bound"]["keep_dyadic_diameter"]
    # Also verify saturated Hoeffding on compressed D=26 raises ValueError instead of returning negative keep/factor
    prob_d26_full = cfb.load_public_rns_problem(
        REPO / "data/rns_inputs/S4_P20_W8.json", max_primes=None, name="S4_P20_W8", spec=spec_r
    )
    _assert_raises(
        lambda: cfb.compute_l1_dyadic_certificate(prob_d26_full, spec_r, trace_upper=None),
        ValueError,
        "compute_l1_dyadic_certificate must reject saturated D=26 Hoeffding instead of negative retention",
    )

    print(
        f"  PASS: EGKMV1 verifier pass + fault injection, shifted_unsigned MGF cert, default L1 Hoeffding cert, "
        f"and live RSA-2048 rolling-ANF R4B_PLUS_COMBINED (2,199 Q, 0 leaked wires, {len(br._free_o)} recycled outcome IDs) verified!"
    )


def main() -> None:
    test_1_exhaustive_stage2_division()
    test_2_universal_cz_phase_synthesis()
    test_3_negative_control_fault_injection()
    test_4_live_stage2_all_19878_primes()
    test_5_live_stage4_crt_and_exact_qq_adders()
    test_6_live_rsa2048_dual_backend()
    test_7_egkmv1_mgf_and_rolling_anf()
    print("\nALL LIVE COMPILER, EGKMV1, MGF, AND DUAL-BACKEND TESTS PASSED!")


if __name__ == "__main__":
    main()
