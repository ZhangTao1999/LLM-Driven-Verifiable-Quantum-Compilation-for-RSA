# Third-Party Notices, Provenance, and Attribution

This release package contains **exclusively original author-written source code, compiled benchmark data, verification receipts, and documentation**. **No third-party source repositories or unmodified upstream files are copied or redistributed inside this package.**

All external papers, published benchmark datasets, wire-stream protocol conventions, and textbook quantum arithmetic algorithms referenced or compared against in our work are explicitly attributed below.

---

## 1. Craig Gidney (2025): Baseline RSA-2048 Architecture & Reference Archive

- **Publication:** C. Gidney, *"How to factor 2048 bit RSA integers with less than a million noisy qubits,"* [arXiv:2505.15917v1](https://arxiv.org/abs/2505.15917v1) (2025).
- **Immutable Zenodo Artifact:** [https://zenodo.org/records/15347487](https://zenodo.org/records/15347487) (`code.zip` SHA-256: `e627abdeb91e880ec8500a3015ab59eb09c3171e1c8f8d9c5eab96728064c94d`).
- **Redistribution Status:** Because no standalone license file (`LICENSE`, `NOTICE`, or `COPYING`) was included in the pinned Zenodo archive, **zero source files from Gidney's archive are redistributed in this repository**.
- **What is included here:** Only our own measured evaluation results (`data/gidney_28_baseline_frontier.json`, `data/accounting_sensitivity.json`) obtained by evaluating Gidney's published architecture across 28 admitted configurations ($w=6$ for $s\in\{3,\dots,14\}\times\ell\in\{20,21\}$, plus $(s,\ell,w)=(14,19,7)$, $(8,21,7)$, $(8,21,8)$, and 64-bit-prep $(2,21,6)$; a 29th attempted point, `C_S2_L21_W6`, failed when 32-bit host precomputation overflowed a bitmask and is logged in `data/accounting_sensitivity.json`), together with our own independent clean-room implementation of Gidney's (2018) ripple-carry adder (`src/flat_stream/gadgets/arith.py`; C. Gidney, *Quantum* **2**, 74 (2018), [arXiv:1709.06648](https://arxiv.org/abs/1709.06648)). Note that Gidney's reference scripts price certain subroutines using macro-level formulas ($b-2$ adder cost, $N-\log_2 N-2$ QROM, $\lfloor\sqrt{K}\rfloor$ phase lookup), which is 1.0%–2.4% more lenient than our strict emitted bit-level gate-stream count (`AND`/`CCX`/`CSWAP`), and Gidney's closed-form analytic qubit estimate omits 36 temporary clean ancillas relative to the 1,323-qubit simulated minimum; our manuscript therefore foregrounds the matched $b-2$ comparison alongside strict bit-level counts and explicit sensitivity audits.
- To independently re-run Gidney's original Python scripts, download `code.zip` directly from [Zenodo 15347487](https://zenodo.org/records/15347487) and verify its SHA-256 digest above.

---

## 2. Gu et al. (2026): `OPSTRM01` Wire-Stream Interface & `uncompute` Convention

- **Publication / Repository:** A. Gu, A. H. Kemeny, S. F. Yelin, and M. D. Lukin, *"Discovering state-of-the-art quantum circuits with large language models,"* [arXiv:2610.22367](https://arxiv.org/abs/2610.22367) / [https://github.com/andigu/uncompute](https://github.com/andigu/uncompute) (Apache-2.0 License).
- **Redistribution Status:** **Zero source files from `andigu/uncompute` are copied or vendored in this repository.**
- **What is included here & compatibility boundary:** `src/flat_stream/` is our own standalone, clean-room Python implementation providing a **Python API-compatible** `Builder`, exact symbolic all-$2^M$ measurement-outcome phase `Simulator`, and O(1)-RAM streaming randomized `kickmix` evaluator (`src/flat_stream/evalcore/`). For lightweight pure-Python local testing, `src/flat_stream/` uses a simplified flat-op serialization with macro opcodes (`AND_COMP`/`AND_REL`/`CSWAP`) rather than the segmented `REGS`/`DATA`/`FLAT`/`LOOP` binary wire encoding of upstream `andigu/uncompute`; consequently, `src/flat_stream/` is **not binary-stream-compatible** with upstream `andigu/uncompute` and does not reproduce cluster `stream_sha256` digests byte-for-byte. All archived full-prime 32-trajectory verification receipts in `receipts/all_prime_32shot/` were generated on cluster hardware by compiling against the independent upstream Apache-2.0 `andigu/uncompute` C kernel together with our independent C parser `Gu_strict_stream_check`. Users who wish to cross-verify against the upstream Apache-2.0 `uncompute` C/Python evaluator may install it directly from `https://github.com/andigu/uncompute`.

---

## 3. Khattar et al. (2026): Randomized Measurement-Phase Verification (`kickmix`)

- **Publication:** T. Khattar, N. C. Rubin, C. Gidney, M. Broughton, G. H. Low, and R. Babbush, *"Leveraging Verifiable Structure in Quantum Circuits to Find More Efficient Compilations with Agentic AI,"* [arXiv:2610.23832](https://arxiv.org/abs/2610.23832) (2026).
- **Redistribution Status:** **Zero external code copied.** Our randomized `kickmix` streaming evaluator (`src/flat_stream/evalcore/__init__.py`) and standalone independent C++17 / Python elementary-gate verifier (`src/elementary_verifier/`, `EGKMV1` format) are independent clean-room implementations of the published algorithm. Note on scope: in local tests (`scripts/run_live_compiler_tests.py`), symbolic (`strict=True`) checks execute through `src/flat_stream.Simulator` while randomized streaming `kickmix` replay executes through the decoupled interpreter `evalcore.StandaloneKickmixEngine` (which does not share `Simulator`'s gate-execution engine), whereas the archived full-prime receipts in `receipts/all_prime_32shot/` were produced by two independent compiled C engines (`Gu_strict_stream_check` + upstream `andigu/uncompute`). The standalone C++17 `EGKMV1` verifier (`src/elementary_verifier/`) covers the unsegmented 13.31B-op odd-CRT `R4B` replay (`receipts/same_rns_controlled_ablation/R4B_unsegmented_elementary_replay.json`), while the newer dyadic-CRT rolling-ANF circuits were certified via segmented `OPSTRM01` streams.

---

## 4. Sharma & Lau (2026): Concurrent Independent RSA-2048 Benchmark Coordinates

- **Publication:** M. Sharma and H. C. Lau, *"Shared Lookups and Lazy Residues: Fewer Toffoli Gates for Factoring RSA-2048,"* [arXiv:2611.08907](https://arxiv.org/abs/2611.08907) / Zenodo [23991044](https://zenodo.org/records/23991044) (2026).
- **Redistribution Status:** **Zero code copied or imported.** `data/sharma_lau_2026_points.json` records only the 13 publicly reported $(Q, \widehat{T}_{\mathrm{fact}})$ literature coordinates from Tables/Figures of Ref. arXiv:2611.08907v1 for direct literature comparison in Fig. 5(b) and Appendix E. Matched-qubit comparisons evaluate both architectures across their shared 1,432–1,806 qubit window (`-24.9%` to `-43.4%`); any floor-to-floor comparison (`-62.8%`) explicitly contrasts our 2,919-qubit minimum-Toffoli endpoint against their 1,806-qubit endpoint at different qubit budgets.

---

## 5. Textbook Quantum Arithmetic & Lookup Primitives Implemented From Published Papers

All quantum subroutines in `src/flat_stream/gadgets/` and `src/compiler/primitives.py` are original clean-room implementations based on the following foundational references:
- **Cuccaro Ripple-Carry Adder:** S. A. Cuccaro, T. G. Draper, S. A. Kutin, and D. P. Moulton, [arXiv:quant-ph/0410184](https://arxiv.org/abs/quant-ph/0410184) (2004).
- **Halving the Cost of Quantum Addition:** C. Gidney, *Quantum* **2**, 74 (2018), [arXiv:1709.06648](https://arxiv.org/abs/1709.06648).
- **Unary-Iteration QROM:** R. Babbush et al., *Phys. Rev. X* **8**, 041015 (2018), [arXiv:1805.03662](https://arxiv.org/abs/1805.03662).
- **SelectSwap QROAM & Clean-Ancilla Phase Lookup (LKS):** G. H. Low, V. Kliuchnikov, and L. Schaeffer, *Quantum* **8**, 1375 (2024), [arXiv:1812.00954](https://arxiv.org/abs/1812.00954).
- **Ekerå–Håstad Short Discrete Logarithms:** M. Ekerå and J. Håstad, *PQCrypto 2017*, [arXiv:1702.00249](https://arxiv.org/abs/1702.00249); M. Ekerå, *Des. Codes Cryptogr.* **88**, 2313–2335 (2020).
- **Residue Number System Modular Exponentiation:** C. Chevignard, P.-A. Fouque, and A. Schrottenloher, *Advances in Cryptology – CRYPTO 2025, Part II*, LNCS Vol. 16001 (Springer, 2025), pp. 384–415, full version ePrint 2024/222.

---

## 6. External Runtime Libraries

Standard Python packages listed in `requirements.txt` (`numpy`, `sympy`, `matplotlib`) are external runtime dependencies installed separately by the user under their respective open-source licenses (BSD / Matplotlib).
