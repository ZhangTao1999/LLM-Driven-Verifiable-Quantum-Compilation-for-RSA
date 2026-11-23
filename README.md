# Reducing RSA-2048 Factoring Resources Below One Billion Toffolis via LLM-Driven Verifiable Quantum Compilation

Open-source code, compiled Pareto frontier ledgers, deterministic RSA-2048 RNS prime inputs, full-prime 32-trajectory verification receipts, exact fixed-point MGF tail certificates, and figure-reproduction scripts accompanying our manuscript submitted to *PRX Quantum*.

> **Provenance:** This repository contains only original author code, data, and verification artifacts. No third-party source repositories are redistributed unattributed (see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for complete citations, DOIs, and upstream provenance, including our attribution to Gu *et al.* (2026) for the `OPSTRM01` binary flat-stream format and C/C++ verifier architecture).

---

## Verified RSA-2048 Modular-Exponentiation Results

Against our 28-circuit sweep of Gidney (2025) under identical clean-ancilla rules and the matched $b-2$ adder convention (Fig. 1), our compiled 30-point strict bit-level Pareto frontier (29 points under matched $b-2$ accounting in Figs. 1 and 3(a), where the 2,822-qubit $f=33$ control is slightly dominated by the 2,485-qubit circuit) lowers the minimum qubit–Toffoli product by **70.1%** (to $2.380\times 10^{12}$ at 2,433 qubits) and expected factoring Toffolis by up to **78.6%** at the same 33-bit accumulator budget (to $0.9608\times 10^9$ at 2,485 qubits, and **79.2%** to $0.9317\times 10^9$ at 2,919 qubits with a 34-bit accumulator), bringing RSA-2048 factoring below $10^9$ Toffoli gates. Under our stricter emitted bit-level gate-stream counting (one Toffoli per emitted logical `AND`, `CCX`, or `CSWAP`, without $\lfloor\sqrt{K}\rfloor$ or $b-2$ macro conventions; Appendix E and Table IV), the reductions remain **69.4%** in product and **78.1%–78.2%** in Toffolis at $f=33$ (**78.8%** at $f=34$), still below $10^9$. The main frontier caps $k \le 24$ to bound classical serialization overhead; an unconstrained $k \in \{26, 30, 36, 40, 42\}$ extension is included in `data/hardware_and_kext_table11.json` and Appendix E Table V:

| Operating Regime | $(s, k, w, D, f)$ | Peak Qubits $Q_{\mathrm{peak}}$ | Matched $b-2$ $\widehat{T}_{\mathrm{fact}}^{\mathrm{SA}}$ | Strict Bit-Level $\widehat{T}_{\mathrm{fact}}$ | Qubit–Toffoli $Q_{\mathrm{peak}}\widehat{T}_{\mathrm{fact}}$ ($b-2$ / Strict) | Verification Tier |
| :--- | :---: | ---: | ---: | ---: | :---: | :--- |
| Min qubits (`S14_PREFIX_COMBINED`) | $(14,1,6,26\mathrm{u},33)$ | **1,306 Q** | $6.2943\times 10^9$ | $6.3548\times 10^9$ | $8.220 / 8.299\times 10^{12}$ | **All-20,288-prime 32-traj.** |
| Low-workspace $k=2$ (`S14_K2_W6_QSAFE`) | $(14,2,6,26\mathrm{u},33)$ | **1,359 Q** | $4.5828\times 10^9$ | $4.6432\times 10^9$ | $6.228 / 6.310\times 10^{12}$ | Batch-0 dual-backend + 5-batch |
| Same-RNS reference (`R4B_PLUS_COMBINED`) | $(4,13,8,26\mathrm{u},33)$ | **2,199 Q** | $1.1289\times 10^9$ | $1.1517\times 10^9$ | $2.482 / 2.533\times 10^{12}$ | **All-19,878-prime 32-traj.** |
| Near-min-$QT$ circuit (`S4_ROLLK20_COMBINED`) | $(4,20,8,26\mathrm{u},33)$ | **2,381 Q** | $1.0003\times 10^9$ | $1.0232\times 10^9$ | $2.382 / 2.436\times 10^{12}$ | **All-19,878-prime 32-traj.** |
| Min $Q_{\mathrm{peak}}\widehat{T}_{\mathrm{fact}}^{\mathrm{SA}}$ (`S4_K22_W8_CMB`) | $(4,22,8,26\mathrm{u},33)$ | **2,433 Q** | **$0.9783\times 10^9$** | $1.0012\times 10^9$ | **$2.380 / 2.436\times 10^{12}$** (`-70.1%` / `-69.4%`) | **All-19,878-prime 32-traj.** |
| Min $f=33$ / first strict $<10^9$ (`S4_K24_W8_OPT`) | $(4,24,8,26\mathrm{u},33)$ | **2,485 Q** | **$0.9608\times 10^9$** (`-78.6%`) | **$0.9836\times 10^9$** (`-78.1%`) | $2.388 / 2.444\times 10^{12}$ | **All-19,878-prime 32-traj.** |
| Wide-window $f=33$ (`S3_K22_W9_F33`) | $(3,22,9,27\mathrm{u},33)$ | **2,822 Q** | $0.9620\times 10^9$ | $0.9799\times 10^9$ | $2.715 / 2.765\times 10^{12}$ | Exact all-prime + Batch-0 dual |
| Wide-window circuit (`S3_K22_W9_COMBINED`) | $(3,22,9,27\mathrm{u},34)$ | **2,823 Q** | $0.9604\times 10^9$ | $0.9782\times 10^9$ | $2.711 / 2.762\times 10^{12}$ | **All-19,682-prime 32-traj.** |
| Min Toffolis (`S3_K24_W9_CMB`) | $(3,24,9,27\mathrm{u},34)$ | **2,919 Q** | **$0.9317\times 10^9$** (`-79.2%`) | **$0.9496\times 10^9$** (`-78.8%`) | $2.720 / 2.772\times 10^{12}$ | Exact all-prime + Batch-0 dual |

- **All-prime 32-trajectory certification (Appendix B Table III):** **6** representative circuits spanning 1,306–2,823 qubits (`119,482` constituent 20-bit prime evaluations across the six circuits, `6,505` clean batch segments, `5,610` zero-segmentation-overhead rolling/bilinear segments including both fixed-$f=33$ sub-billion milestones, `3.14 TB` of unique bit-level binary streams replayed across 16 primary + 16 seed-isolated holdout trajectories = `6.28 TB` total replayed with 100% identical per-segment SHA-256 digests).
- **Exact all-prime monolithic opcode profiles (Appendix E Table Va):** Zero-sampling all-prime counts (`AND`/`CCX`/`CSWAP`, `CX`, `CZ`, `HMR`) across all 6 all-prime certified circuits (`diff == 0` vs. certified `T_mono`), plus exact all-prime monolithic counts on `S3_K22_W9_F33` and `S3_K24_W9_CMB`.
- **Same-RNS controlled ablation (`19,878` primes, $f=33$):** Comparing our 16-trajectory all-prime truncated-modulus flat-stream receipt (`2,204` qubits, `246.97M` Toffolis/run = `12,424.4` Toffolis/prime, `3.23M` segmentation overhead) against our dyadic rolling-ANF circuit (`2,199` qubits, `227.41M` Toffolis/run = `11,440.5` Toffolis/prime, `0` segmentation overhead) isolates `-5` qubits, `-983.8` Toffolis/prime (`-7.92%` per run; `-8.14%` in $\widehat{T}_{\mathrm{fact}}$), and complete removal of verification-boundary overhead.

---

## Quickstart Verification (4 Commands)

Requires Python $\ge 3.10$ (standard library only for Step 1; `pip install -r requirements.txt` for Steps 2–4).

```bash
# 1. Offline SHA-256, RNS prime audit, Prop. 3 dyadic bound, 6 all-prime receipts, MGF/Hoeffding & Table V audit (~30-120 s, stdlib only)
python scripts/verify_all_paper_claims.py

# 2. Live compiler, exhaustive small-width checks, strict/non-strict fault injection, EGKMV1 & RSA-2048 dual-backend check (~30-90 s)
python scripts/run_live_compiler_tests.py

# 3. Compile & profile any of the 49 frontier or 5 k>24 extension configs directly from data/rns_inputs/ (~15 s - 5 min depending on k)
python scripts/profile_frontier_config.py --list
python scripts/profile_frontier_config.py --id S8_K3_W8_OPT
python scripts/profile_frontier_config.py --id S4_K22_W8_CMB
python scripts/profile_frontier_config.py --id S4_K26_W8_EXT

# 4. Reproduce manuscript data Figs. 1, 3, and 5 (plus supplemental convention comparison) into output_figures/ (~5 s)
python scripts/reproduce_paper_figures.py
```

> **Note on Archived Receipt Provenance & Verifier Scope:** See [`receipts/README.md`](receipts/README.md) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for full details on (i) authoritative `v3` `l1_dyadic_certificate.json` Proposition 3 retention bounds vs. historical `v1` metadata inside `receipt.json`, (ii) exact SHA-256 and canonical prime-order binding between archived cluster receipts and `data/rns_inputs/*.json`, and (iii) the distinction between archived 3.14 TB cluster C-backend receipts (pairing cited Apache-2.0 `andigu/uncompute` C `kickmix` with our cluster C symbolic parser, not redistributed here) and the self-contained local dual-engine verification shipped in `src/flat_stream/` (`Simulator` symbolic all-$2^M$ engine + decoupled `evalcore.StandaloneKickmixEngine` streaming `OPSTRM01` interpreter).

---

## Repository Structure

```text
.
├── .gitattributes                             # Pins LF line endings for cross-platform SHA-256 consistency
├── README.md                                  # This guide
├── LICENSE                                    # MIT License
├── THIRD_PARTY_NOTICES.md                     # Complete third-party provenance & algorithm citations
├── requirements.txt                           # Minimal Python dependencies (numpy, sympy, matplotlib)
├── MANIFEST_SHA256.json                       # SHA-256 digest of every file in this release
├── src/
│   ├── flat_stream/                           # Clean-room Python Builder + symbolic all-2^M Simulator + decoupled evalcore.StandaloneKickmixEngine, adders & QROM
│   ├── compiler/                              # Core RSA-2048 batched RNS + dyadic torus mod-2^f compiler (primitives.py, bridge.py)
│   ├── elementary_verifier/                   # Archived standalone C++17 & Python elementary-gate verifier (EGKMV1, legacy odd-CRT provenance)
│   └── mgf_certificates/                      # Exact fixed-point C++ MGF moment worker, App. C.6 Hoeffding & 256-bit Chernoff verifier
├── data/
│   ├── combined_frontier_49_configs.json      # Complete 49-circuit compiled dataset (embedded specs) & 30-point Pareto frontier (Table IV)
│   ├── hardware_and_kext_table11.json         # Exact all-prime 6-circuit hardware profiles & 5 unconstrained k > 24 extensions (Table V)
│   ├── gidney_28_baseline_frontier.json       # Complete 28-circuit Gidney (2025) baseline sweep & 25-point Pareto frontier (App. A.5)
│   ├── macro_ablation_89_circuits.json        # Archived early-development 89-circuit macro-level ablation dataset
│   ├── accounting_sensitivity.json            # Closed-form Gidney analytic vs. simulated sensitivity checks (App. E Fig. 5a)
│   ├── sharma_lau_2026_points.json            # Published 13 RSA-2048 points from Sharma & Lau (2026) (App. E Fig. 5b)
│   └── rns_inputs/                            # All 9 deterministic RSA-2048 RNS prime-set inputs
├── receipts/
│   ├── README.md                              # Schema & provenance guide for historical receipts vs. authoritative v3 certificates
│   ├── all_prime_32shot/                      # Primary + 16-shot holdout receipts & rigorous Prop. 3 L1 dyadic certificates for all 6 circuits
│   ├── same_rns_controlled_ablation/          # Same-RNS (19,878p, f=33) odd-CRT receipt & archived unsegmented elementary-gate replay receipt
│   └── mgf_tail_certificates/                 # All 23 archived full-prime MGF tail-bound certificates (App. C.6)
└── scripts/
    ├── verify_all_paper_claims.py             # Offline SHA-256 + RNS prime audit + Prop. 3 + 6 receipts + MGF/Hoeffding + Table V audit
    ├── run_live_compiler_tests.py             # Live exhaustive primitive, fault-injection, EGKMV1, Stage-(ii)/(iv) & RSA-2048 dual-backend tests
    ├── profile_frontier_config.py             # Re-compile and profile any of the 49 frontier or 5 k>24 extension configs on demand
    └── reproduce_paper_figures.py             # Regenerate manuscript data Figs. 1, 3, and 5 (plus convention comparison) directly from data/
```

---

## Optional C++ Compilation

All end-to-end verification and profiling workflows in this release run through the four Python scripts above (including the pure-Python `EGKMV1` verifier and stdlib/fixed-point MGF checks). The C++17 sources below are the standalone high-throughput cluster reference engines used for the archived unsegmented 13.31-billion-op legacy odd-CRT replay and full-family 977M-entry MGF moment enumeration (binary stream protocols are documented at the top of each source file):

- **Archived Standalone Elementary Verifier (`src/elementary_verifier/`, portable ISO C++17):**
  ```bash
  g++ -O3 -std=c++17 src/elementary_verifier/elementary_kickmix_verifier.cpp -o elementary_kickmix_verifier
  g++ -O3 -std=c++17 src/elementary_verifier/macro_stream_lowerer.cpp -o macro_stream_lowerer
  ```
- **Fixed-Point MGF Moment Worker (`src/mgf_certificates/`, requires 64-bit GCC/Clang for `unsigned __int128`, OpenMP, and OpenSSL `libcrypto`; add `-Wno-deprecated-declarations` on OpenSSL 3.x):**
  ```bash
  g++ -O3 -std=c++17 -fopenmp -Wno-deprecated-declarations src/mgf_certificates/support_worker.cpp -lcrypto -o support_worker
  ```

---

## Authors, AI Usage Disclosure, and Citation

- **Authors:** Xiaodi Li\*, Yuqing Wang\*, Xue Chen, Yang-Yang Tang, Linghui Chen† (`chenlh@mail.ustc.edu.cn`), and Tao Zhang‡ (`tao_zhang@mail.tsinghua.edu.cn`).
  - \* Equal contribution. † Corresponding author (`chenlh@mail.ustc.edu.cn`). ‡ Corresponding author (`tao_zhang@mail.tsinghua.edu.cn`).
- **AI Usage Disclosure & Scope of Release:** The core architectural discoveries in the accompanying manuscript were generated using GPT-6 Astra (2026) within the researcher-frozen verification workflow described in Appendix B, and AI tools assisted with manuscript preparation. While the automated LLM search harness used during discovery is not part of this open-source release, full reproducibility rests entirely on the deterministic open-source compiler, independent verifiers, RNS prime inputs, cryptographic receipts, and exact tail certificates provided here.
- **Citation:**
  ```bibtex
  @article{li2026subbillion,
    title   = {Reducing {RSA}-2048 Factoring Resources Below One Billion Toffolis via {LLM}-Driven Verifiable Quantum Compilation},
    author  = {Li, Xiaodi and Wang, Yuqing and Chen, Xue and Tang, Yang-Yang and Chen, Linghui and Zhang, Tao},
    year    = {2026},
    journal = {arXiv preprint}
  }
  ```
