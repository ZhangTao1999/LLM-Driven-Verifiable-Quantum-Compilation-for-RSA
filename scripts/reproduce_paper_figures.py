#!/usr/bin/env python3
"""Reproduce manuscript data figures (Figs. 1, 3, and 5, plus supplemental convention comparison) directly from data/ into output_figures/.

- Fig. 1 (fig1_pareto_main): Main RSA-2048 Pareto frontier under Gidney's matched b-2 adder
  convention T_fact^SA (29 Pareto-optimal points from our 30-circuit strict bit-level frontier;
  -70.1% min Q_peak*T_fact^SA at 2,433 Q; -78.6% at f=33; -79.2% at f=34).
- Fig. 3 (fig3_spacetime_and_waterfall): 1x2 figure combining (a) qubit-Toffoli product
  Q_peak*T_fact^SA vs. baseline and (b) all-19,878-prime per-prime emitted bit-level Toffoli
  waterfall at (s=4, ell=20, w=8, f=33) from truncated-modulus bit-level replay through rolling k=24.
- Supplemental (supp_convention_compare): Direct comparison of strict bit-level Toffoli T_fact
  vs. Gidney-matched b-2 adder T_fact^SA across all 30 Pareto circuits (Appendix E & Table IV).
- Fig. 5 (fig5_accounting_and_concurrent): Appendix E Gidney (2025) analytic sensitivity and
  comparison against concurrent work (Sharma & Lau, 2026) under strict bit-level Toffoli counting.

Usage:
    python scripts/reproduce_paper_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Python >= 3.10 is required to run this figure-reproduction script.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "output_figures"


def load_json(rel: str):
    return json.loads((REPO / rel).read_text(encoding="utf-8"))


def pareto_clean(pts):
    out, best = [], float("inf")
    for p in sorted(pts, key=lambda x: (x["Q"], x["T_fact"], x.get("point", ""))):
        if p["T_fact"] < best:
            out.append(p)
            best = p["T_fact"]
    return out


def best_at_cap(pts, cap):
    feas = [p for p in pts if p["Q"] <= cap]
    return min(feas, key=lambda p: p["T_fact"]) if feas else None


def plot_step(ax, pts, label, color, ls="-", lw=1.5, ms=12, alpha=1.0, zorder=3, extend_to=None, marker="o"):
    qs = [p["Q"] for p in pts]
    ts = [p["T_fact"] / 1e9 for p in pts]
    qs_step = qs + [extend_to] if (extend_to and extend_to > qs[-1]) else qs
    ts_step = ts + [ts[-1]] if (extend_to and extend_to > qs[-1]) else ts
    ax.step(qs_step, ts_step, where="post", color=color, linestyle=ls, linewidth=lw, alpha=alpha, label=label, zorder=zorder)
    ax.scatter(qs, ts, s=ms, marker=marker, color=color, alpha=alpha, zorder=zorder + 0.2, edgecolors="none")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": 8.4,
        "axes.titlesize": 8.8,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.75,
    })

    m = load_json("data/combined_frontier_49_configs.json")
    g28 = load_json("data/gidney_28_baseline_frontier.json")
    comp_acc = load_json("data/accounting_sensitivity.json")
    sl_pts = sorted(load_json("data/sharma_lau_2026_points.json")["points"], key=lambda x: x["Q"])

    by = {r["id"]: r for r in m["all_configs"]}
    cmb_pareto_all = [by[pid] for pid in m["pareto_ids_all"]]
    cmb_pts_strict = [{"point": r["id"], "Q": r["Q_peak"], "T_fact": r["T_fact_strict"], "QT": r["QT_strict"]} for r in cmb_pareto_all]
    cmb_pts_sa_p = pareto_clean([{"point": r["id"], "Q": r["Q_peak"], "T_fact": r["T_fact_same_adder"], "QT": r["QT_same_adder"]} for r in cmb_pareto_all])

    ref_pts = pareto_clean([
        {"point": p.get("point", ""), "Q": int(p["dual_meter"]["original_common_meter"]["Q"]), "T_fact": float(p["dual_meter"]["original_common_meter"]["T_fact"])}
        for p in g28["canonical_reference_28"]
    ])
    ref_min_qt_pt = min(ref_pts, key=lambda p: p["Q"] * p["T_fact"])
    ref_min_qt = ref_min_qt_pt["Q"] * ref_min_qt_pt["T_fact"]
    ref_min_t = min(p["T_fact"] for p in ref_pts)
    r4bp_rec = load_json("receipts/same_rns_controlled_ablation/R4B_PLUS_odd_crt_receipt.json")
    r4bp_t_shot = float(r4bp_rec["measured_resources"]["toffoli_monolithic_equivalent"])
    r4b_primes = float(r4bp_rec["public_input"]["prime_count"])

    CMB_MIN_Q = by["S14_PREFIX_COMBINED"]
    CMB_MIN_QT = by["S4_K22_W8_CMB"]
    CMB_MIN_T_ALL = by["S3_K24_W9_CMB"]
    CMB_R4B = by["R4B_PLUS_COMBINED"]
    CMB_K20 = by["S4_ROLLK20_COMBINED"]
    CMB_K24 = by["S4_K24_W8_OPT"]

    GIDNEY_SIM_Q = g28["gidney_paper_config_simulated"]["Q_peak"]
    GIDNEY_SIM_T = g28["gidney_paper_config_simulated"]["T_fact"] / 1e9
    ann_box = dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#e2e8f0", linewidth=0.45, alpha=0.95)
    comma_fmt = mticker.FuncFormatter(lambda x, _: f"{int(round(x)):,}")

    # ---------------- Fig. 1: Main Pareto Frontier (Gidney-matched b-2 adder) ----------------
    qt_sa_red = 100.0 * (1.0 - CMB_MIN_QT["QT_same_adder"] / ref_min_qt)
    t_f33_sa_red = 100.0 * (1.0 - CMB_K24["T_fact_same_adder"] / ref_min_t)
    t_all_sa_red = 100.0 * (1.0 - CMB_MIN_T_ALL["T_fact_same_adder"] / ref_min_t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.15, 3.22))
    plot_step(ax1, ref_pts, "Gidney (2025) baseline", "#64748b", ls="--", lw=1.5, ms=13, zorder=3, extend_to=2930)
    ax1.scatter([GIDNEY_SIM_Q], [GIDNEY_SIM_T], marker="^", s=38, color="#1e293b", zorder=7, label="Gidney (2025) paper config")
    plot_step(ax1, cmb_pts_sa_p, "This work", "#047857", ls="-", lw=2.1, ms=17, zorder=6)

    q_grid = np.arange(1323, 2920)
    ref_env = np.array([best_at_cap(ref_pts, q)["T_fact"] / 1e9 for q in q_grid])
    cmb_env_sa = np.array([best_at_cap(cmb_pts_sa_p, q)["T_fact"] / 1e9 for q in q_grid])
    ax1.fill_between(q_grid, cmb_env_sa, ref_env, color="#047857", alpha=0.10, zorder=1)
    ax1.axhline(1.0, color="#475569", ls=":", lw=1.05, zorder=5)

    ax1.scatter([ref_min_qt_pt["Q"]], [ref_min_qt_pt["T_fact"] / 1e9], marker="s", s=24, color="#334155", zorder=7)
    ax1.annotate(f"Baseline min $Q_{{\\mathrm{{peak}}}}\\hat{{T}}_{{\\mathrm{{fact}}}}$\n{ref_min_qt_pt['Q']:,} Q, ${ref_min_qt_pt['T_fact']/1e9:.2f}\\times 10^9$",
                 (ref_min_qt_pt["Q"], ref_min_qt_pt["T_fact"] / 1e9), xytext=(1905, 5.55), bbox=ann_box, fontsize=5.9, color="#334155",
                 arrowprops=dict(arrowstyle="->", lw=0.6, color="#475569"))
    min_q_sa = CMB_MIN_Q["T_fact_same_adder"] / 1e9
    ax1.scatter([CMB_MIN_Q["Q_peak"]], [min_q_sa], marker="o", s=44, color="#047857", edgecolor="white", lw=0.75, zorder=8)
    ax1.annotate("Min qubits: 1,306 Q", (CMB_MIN_Q["Q_peak"], min_q_sa),
                 xytext=(1295, 1.32), bbox=ann_box, fontsize=6.0, color="#047857",
                 arrowprops=dict(arrowstyle="->", lw=0.65, color="#047857"))
    min_qt_sa_t = CMB_MIN_QT["T_fact_same_adder"] / 1e9
    ax1.scatter([CMB_MIN_QT["Q_peak"]], [min_qt_sa_t], marker="o", s=54, color="#9f1239", edgecolor="white", lw=0.8, zorder=9)
    ax1.annotate(f"Min $Q_{{\\mathrm{{peak}}}}\\hat{{T}}_{{\\mathrm{{fact}}}}^{{\\mathrm{{SA}}}}$ ($-{qt_sa_red:.1f}\\%$)",
                 (CMB_MIN_QT["Q_peak"], min_qt_sa_t), xytext=(1835, 2.75), bbox=ann_box, fontsize=6.0, color="#9f1239",
                 arrowprops=dict(arrowstyle="->", lw=0.75, color="#9f1239"))
    min_t_sa = CMB_MIN_T_ALL["T_fact_same_adder"] / 1e9
    ax1.scatter([CMB_MIN_T_ALL["Q_peak"]], [min_t_sa], marker="o", s=44, color="#047857", edgecolor="white", lw=0.75, zorder=8)
    ax1.annotate(f"Min Toffolis\n${min_t_sa:.4f}\\times 10^9$ ($-{t_all_sa_red:.1f}\\%$)",
                 (CMB_MIN_T_ALL["Q_peak"], min_t_sa), xytext=(2285, 1.62), bbox=ann_box, fontsize=5.95, color="#047857",
                 arrowprops=dict(arrowstyle="->", lw=0.65, color="#047857"))

    ax1.set_yscale("log")
    ax1.set_xlim(1275, 2975)
    ax1.set_ylim(0.80, 13.5)
    ax1.set_yticks([1.0, 1.5, 2, 3, 4, 6, 8, 11])
    ax1.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
    ax1.xaxis.set_major_formatter(comma_fmt)
    ax1.set_xlabel(r"Peak logical qubits $Q_{\mathrm{peak}}$")
    ax1.set_ylabel(r"Expected Toffolis $\hat{T}_{\mathrm{fact}}^{\mathrm{SA}}$ ($10^9$)")
    ax1.set_title("(a) Full RSA-2048 Pareto frontier", pad=5)
    ax1.grid(True, which="both", alpha=0.22, lw=0.45)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#cbd5e1", fontsize=5.8)

    sa_z_main = [p for p in cmb_pts_sa_p if p["Q"] >= 1840]
    plot_step(ax2, sa_z_main, None, "#047857", ls="-", lw=2.1, ms=18, zorder=6)
    ax2.axhline(1.0, color="#475569", ls=":", lw=1.05, zorder=4)
    t_2433_sa = CMB_MIN_QT["T_fact_same_adder"] / 1e9
    t_2485_sa = CMB_K24["T_fact_same_adder"] / 1e9
    ax2.scatter([2433], [t_2433_sa], marker="o", s=62, color="#9f1239", edgecolor="white", lw=0.85, zorder=10)
    ax2.scatter([2485], [t_2485_sa], marker="o", s=46, color="#0f766e", edgecolor="white", lw=0.80, zorder=9)
    ax2.scatter([2919], [min_t_sa], marker="o", s=50, color="#047857", edgecolor="white", lw=0.85, zorder=9)
    ax2.annotate(f"Min $Q_{{\\mathrm{{peak}}}}\\hat{{T}}_{{\\mathrm{{fact}}}}^{{\\mathrm{{SA}}}}$ & first $<10^9$\n2,433 Q, ${t_2433_sa:.4f}\\times 10^9$",
                 (2433, t_2433_sa), xytext=(2315, 1.175), bbox=ann_box, fontsize=5.85, color="#9f1239",
                 arrowprops=dict(arrowstyle="->", lw=0.72, color="#9f1239"))
    ax2.annotate(f"Min $f{{=}}33$ ($-{t_f33_sa_red:.1f}\\%$)\n2,485 Q, ${t_2485_sa:.4f}\\times 10^9$",
                 (2485, t_2485_sa), xytext=(1915, 0.892), bbox=ann_box, fontsize=5.80, color="#0f766e",
                 arrowprops=dict(arrowstyle="->", lw=0.65, color="#0f766e"))
    ax2.annotate(f"Min Toffolis ($f{{=}}34$)\n${min_t_sa:.4f}\\times 10^9$",
                 (2919, min_t_sa), xytext=(2495, 0.888), bbox=ann_box, fontsize=5.85, color="#047857",
                 arrowprops=dict(arrowstyle="->", lw=0.65, color="#047857"))
    ax2.set_xlim(1865, 2970)
    ax2.set_ylim(0.860, 1.485)
    ax2.xaxis.set_major_formatter(comma_fmt)
    ax2.set_xlabel(r"Peak logical qubits $Q_{\mathrm{peak}}$")
    ax2.set_ylabel(r"Expected Toffolis $\hat{T}_{\mathrm{fact}}^{\mathrm{SA}}$ ($10^9$)")
    ax2.set_title("(b) Minimum-product & sub-billion zoom", pad=5)
    ax2.grid(True, alpha=0.22, lw=0.45)

    fig.tight_layout(pad=0.48, w_pad=1.05)
    fig.savefig(OUT_DIR / "fig1_pareto_main.pdf")
    fig.savefig(OUT_DIR / "fig1_pareto_main.png", dpi=240)
    plt.close(fig)

    # ---------------- Fig. 3 (1x2): Space-Time Product & All-Prime Per-Prime Waterfall ----------------
    cmb_qt_sa_red = 100.0 * (1.0 - CMB_MIN_QT["QT_same_adder"] / ref_min_qt)
    cmb_qt_sa_val = CMB_MIN_QT["QT_same_adder"] / 1e12

    fig, (ax_qt, ax_wf) = plt.subplots(1, 2, figsize=(7.15, 2.95))

    ax_qt.plot([p["Q"] for p in ref_pts], [p["Q"] * p["T_fact"] / 1e12 for p in ref_pts], "--o", color="#64748b", lw=1.4, ms=3.0, label=f"Gidney (2025) baseline (min {ref_min_qt/1e12:.2f})")
    ax_qt.plot([p["Q"] for p in cmb_pts_sa_p], [p["QT"] / 1e12 for p in cmb_pts_sa_p], "-D", color="#047857", lw=1.9, ms=3.4, label=f"This work (min {cmb_qt_sa_val:.3f})")
    ax_qt.scatter([CMB_MIN_QT["Q_peak"]], [cmb_qt_sa_val], marker="*", s=110, color="#9f1239", edgecolor="white", lw=0.45, zorder=9)
    ax_qt.annotate(f"Global min $Q_{{\\mathrm{{peak}}}}\\hat{{T}}_{{\\mathrm{{fact}}}}^{{\\mathrm{{SA}}}}$ ($s{{=}}4, k{{=}}22$)\n2,433 Q, ${cmb_qt_sa_val:.3f}\\times 10^{{12}}$ ($-{cmb_qt_sa_red:.1f}\\%$)",
                   (CMB_MIN_QT["Q_peak"], cmb_qt_sa_val), xytext=(1755, 5.25), bbox=ann_box, fontsize=5.8, color="#9f1239", fontweight="bold",
                   arrowprops=dict(arrowstyle="->", lw=0.7, color="#9f1239"))
    ax_qt.set_xlim(1275, 2970)
    ax_qt.set_ylim(1.5, 15.3)
    ax_qt.xaxis.set_major_formatter(comma_fmt)
    ax_qt.set_xlabel(r"Peak logical qubits $Q_{\mathrm{peak}}$")
    ax_qt.set_ylabel(r"Qubit–Toffoli product $Q_{\mathrm{peak}}\hat{T}_{\mathrm{fact}}^{\mathrm{SA}}$ ($10^{12}$)")
    ax_qt.set_title(r"(a) Qubit–Toffoli product $Q_{\mathrm{peak}}\hat{T}_{\mathrm{fact}}^{\mathrm{SA}}$", pad=5, fontsize=8.3)
    ax_qt.grid(True, alpha=0.22, lw=0.45)
    ax_qt.legend(loc="upper right", fontsize=5.65, frameon=True, facecolor="white")

    # 5-bar all-19,878-prime bit-level progression (zero legacy macro-era bar)
    wf_labels = ["Trunc.-CRT\nbit-level\n($k{=}13$)", "Dyadic, (ii),\nroll. ($k{=}13$)", "$k{=}20$\n(rolling)", "$k{=}22$\n(min prod.)", "$k{=}24$\n($f{=}33$)"]
    wf_exact = [
        r4bp_t_shot / r4b_primes,
        float(CMB_R4B["per_prime_total"]),
        float(CMB_K20["per_prime_total"]),
        float(CMB_MIN_QT["per_prime_total"]),
        float(CMB_K24["per_prime_total"]),
    ]
    wf_vals = [round(v) for v in wf_exact]
    wf_cols = ["#3b82f6", "#10b981", "#059669", "#9f1239", "#047857"]
    xwf = np.arange(len(wf_vals))
    ax_wf.bar(xwf, wf_vals, width=0.56, color=wf_cols, alpha=0.90, edgecolor="#1e293b", linewidth=0.55)
    for i, (xv, val) in enumerate(zip(xwf, wf_vals)):
        ax_wf.text(xv, val + 105, f"{val:,}", ha="center", va="bottom", fontsize=5.95, fontweight="bold" if i in (0, 1, 3) else "normal", color="#0f172a")
        if i > 0:
            delta = round(wf_exact[i] - wf_exact[i - 1])
            ax_wf.plot([xv - 0.72, xv - 0.28], [wf_vals[i - 1], wf_vals[i - 1]], "--", color="#64748b", lw=0.65)
            d_txt = f"{abs(delta):,}".replace(",", "{,}")
            ax_wf.text(xv, val - 280, f"$-{d_txt}$", ha="center", va="top", fontsize=5.55, color="white", fontweight="bold")
    ax_wf.set_xticks(xwf)
    ax_wf.set_xticklabels(wf_labels, fontsize=5.45)
    ax_wf.set_ylim(8800, 13100)
    ax_wf.yaxis.set_major_formatter(comma_fmt)
    d_brk = 0.018
    ax_wf.plot((-d_brk, +d_brk), (0.04 - d_brk, 0.04 + d_brk), transform=ax_wf.transAxes, color="#1e293b", lw=0.85, clip_on=False)
    ax_wf.plot((-d_brk, +d_brk), (0.078 - d_brk, 0.078 + d_brk), transform=ax_wf.transAxes, color="#1e293b", lw=0.85, clip_on=False)
    ax_wf.set_ylabel("Toffolis per prime (bit-level)")
    ax_wf.set_title(r"(b) All-prime progression ($s{=}4, \ell{=}20, w{=}8$)", pad=5, fontsize=8.3)
    ax_wf.grid(axis="y", alpha=0.22, lw=0.45)

    fig.tight_layout(pad=0.48, w_pad=1.05)
    fig.savefig(OUT_DIR / "fig3_spacetime_and_waterfall.pdf")
    fig.savefig(OUT_DIR / "fig3_spacetime_and_waterfall.png", dpi=240)
    plt.close(fig)

    # ---------------- Supplemental (archived convention comparison): Matched b-2 vs Strict Bit-Level Toffoli ----------------
    fig, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(7.15, 3.05))
    cmb_st_z = [p for p in cmb_pts_strict if p["Q"] >= 1840]
    sa_z = [p for p in cmb_pts_sa_p if p["Q"] >= 1840]
    plot_step(ax_c1, cmb_st_z, "Strict bit-level Toffoli (quantum add $b{-}1$)", "#047857", ls="-", lw=2.0, ms=15, zorder=6)
    plot_step(ax_c1, sa_z, "Gidney-matched adder ($b{-}2$)", "#1d4ed8", ls="--", lw=1.65, ms=13, zorder=5)
    ax_c1.axhline(1.0, color="#475569", ls=":", lw=0.95, zorder=4)
    q_zgrid = np.arange(1893, 2920)
    ax_c1.fill_between(q_zgrid, [best_at_cap(cmb_pts_sa_p, q)["T_fact"] / 1e9 for q in q_zgrid],
                       [best_at_cap(cmb_pts_strict, q)["T_fact"] / 1e9 for q in q_zgrid], color="#3b82f6", alpha=0.14, zorder=2)
    sa_2433, st_2433 = CMB_MIN_QT["T_fact_same_adder"] / 1e9, CMB_MIN_QT["T_fact_strict"] / 1e9
    sa_2485, st_2485 = CMB_K24["T_fact_same_adder"] / 1e9, CMB_K24["T_fact_strict"] / 1e9
    sa_2919, st_2919 = CMB_MIN_T_ALL["T_fact_same_adder"] / 1e9, CMB_MIN_T_ALL["T_fact_strict"] / 1e9
    ax_c1.annotate(f"2,433 Q (min $Q_{{\\mathrm{{peak}}}}\\hat{{T}}_{{\\mathrm{{fact}}}}$)\nstrict: ${st_2433:.4f}$ | $b{{-}}2$: ${sa_2433:.4f}$ ($\\times 10^9$)",
                   (2433, st_2433), xytext=(2335, 1.185), bbox=ann_box, fontsize=5.35, color="#9f1239", arrowprops=dict(arrowstyle="->", lw=0.65, color="#9f1239"))
    ax_c1.annotate(f"2,485 Q ($f{{=}}33$)\nstrict: ${st_2485:.4f}$ | $b{{-}}2$: ${sa_2485:.4f}$",
                   (2485, st_2485), xytext=(1888, 0.892), bbox=ann_box, fontsize=5.35, color="#047857", arrowprops=dict(arrowstyle="->", lw=0.6, color="#047857"))
    ax_c1.annotate(f"2,919 Q (min Toffolis)\nstrict: ${st_2919:.4f}$ | $b{{-}}2$: ${sa_2919:.4f}$",
                   (2919, sa_2919), xytext=(2385, 0.880), bbox=ann_box, fontsize=5.35, color="#1d4ed8", arrowprops=dict(arrowstyle="->", lw=0.65, color="#1d4ed8"))
    ax_c1.set_xlim(1865, 2970)
    ax_c1.set_ylim(0.860, 1.460)
    ax_c1.xaxis.set_major_formatter(comma_fmt)
    ax_c1.set_xlabel(r"Peak logical qubits $Q_{\mathrm{peak}}$")
    ax_c1.set_ylabel(r"Expected Toffolis ($10^9$)")
    ax_c1.set_title("(a) Strict bit-level Toffoli vs. matched $b{-}2$ frontier", pad=5, fontsize=8.3)
    ax_c1.grid(True, alpha=0.22, lw=0.45)
    ax_c1.legend(loc="upper right", fontsize=5.65, frameon=True, facecolor="white")

    for w_val, mk, col, nm in [(6, "s", "#0d9488", "$w{=}6$ low-workspace (Stage-(i) clean-carry add)"), (8, "o", "#1d4ed8", "$w{=}8$ packed & rolling ANF"), (9, "D", "#7c3aed", "$w{=}9$ wide-window")]:
        pts_w = [r for r in cmb_pareto_all if r["w"] == w_val]
        ds = [(r["T_fact_strict"] / r["T_fact_same_adder"] - 1.0) * 100.0 for r in pts_w]
        ax_c2.scatter([r["Q_peak"] for r in pts_w], ds, marker=mk, s=28, color=col, edgecolor="white", lw=0.35, zorder=6, label=f"{nm} ({min(ds):.2f}–{max(ds):.2f}%)")
    all_d = [(r["T_fact_strict"] / r["T_fact_same_adder"] - 1.0) * 100.0 for r in cmb_pareto_all]
    ax_c2.axhspan(min(all_d), max(all_d), color="#e2e8f0", alpha=0.40, zorder=1)
    ax_c2.set_xlim(1270, 2980)
    ax_c2.set_ylim(0.55, 2.75)
    ax_c2.xaxis.set_major_formatter(comma_fmt)
    ax_c2.set_xlabel(r"Peak logical qubits $Q_{\mathrm{peak}}$")
    ax_c2.set_ylabel(r"Strict bit-level Toffoli vs. $b{-}2$ gap (%)")
    ax_c2.set_title("(b) Exact LSB-carry gap across all 30 circuits", pad=5, fontsize=8.3)
    ax_c2.grid(True, alpha=0.22, lw=0.45)
    ax_c2.legend(loc="lower right", fontsize=5.55, frameon=True, facecolor="white")

    fig.tight_layout(pad=0.48, w_pad=1.05)
    fig.savefig(OUT_DIR / "supp_convention_compare.pdf")
    fig.savefig(OUT_DIR / "supp_convention_compare.png", dpi=240)
    plt.close(fig)

    # ---------------- Fig. 5 (Appendix E): Analytic Sensitivity & Sharma-Lau (2026) ----------------
    a_fix_ref = comp_acc["frontiers"]["historical_A_mixed"]["reference"]
    a_free_ref = comp_acc["frontiers"]["historical_A_free_mixed"]["reference"]
    a_free_min_qt = min(p["Q"] * p["T_fact"] for p in a_free_ref)
    red_vs_free_f = 100.0 * (1.0 - CMB_MIN_QT["QT_strict"] / a_free_min_qt)
    sl_reds = [100.0 * (1.0 - best_at_cap(cmb_pts_strict, p["Q"])["T_fact"] / p["T_fact"]) for p in sl_pts]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.15, 3.05))
    plot_step(ax_a, ref_pts, "Gidney (2025) 28 sim. baselines", "#94a3b8", ls="-.", lw=1.25, ms=8, zorder=2, extend_to=2930)
    plot_step(ax_a, a_free_ref, r"Gidney analytic free ($f\in[24,59]$)", "#64748b", ls=":", lw=1.35, ms=7, zorder=3, extend_to=2930)
    plot_step(ax_a, a_fix_ref, r"Gidney analytic fixed ($f{=}33, b_{\mathrm{mask}}{=}26$)", "#334155", ls="--", lw=1.4, ms=9, zorder=4, extend_to=2930)
    plot_step(ax_a, cmb_pts_strict, f"This work, strict ($-{red_vs_free_f:.1f}\\%$ min $Q\\hat{{T}}$ vs. free $f$)", "#047857", ls="-", lw=2.0, ms=16, zorder=6)
    ax_a.axhline(1.0, color="#475569", ls=":", lw=0.95)
    ax_a.set_xlim(1235, 2975)
    ax_a.set_ylim(0.75, 14.8)
    ax_a.xaxis.set_major_formatter(comma_fmt)
    ax_a.set_xlabel(r"Peak logical qubits $Q_{\mathrm{peak}}$")
    ax_a.set_ylabel(r"Expected Toffolis $\hat{T}_{\mathrm{fact}}$ ($10^9$)")
    ax_a.set_title("(a) Gidney (2025) simulated & analytic sensitivity", pad=5, fontsize=8.3)
    ax_a.grid(True, alpha=0.22, lw=0.45)
    ax_a.legend(loc="upper right", fontsize=5.2, frameon=True, facecolor="white")

    plot_step(ax_b, ref_pts, "Gidney (2025) baseline", "#94a3b8", ls=":", lw=1.35, ms=10, zorder=2, extend_to=2930)
    plot_step(ax_b, sl_pts, "Sharma & Lau (2026) (13 pts)", "#d95f02", ls="--", lw=1.75, ms=18, zorder=5, extend_to=2930)
    plot_step(ax_b, cmb_pts_strict, f"This work, strict ($-{min(sl_reds):.1f}\\%$ to $-{max(sl_reds):.1f}\\%$ matched $Q$)", "#047857", ls="-", lw=2.0, ms=16, zorder=6)
    ax_b.axhline(1.0, color="#475569", ls=":", lw=0.95)
    ax_b.set_xlim(1235, 2975)
    ax_b.set_ylim(0.75, 11.8)
    ax_b.xaxis.set_major_formatter(comma_fmt)
    ax_b.set_xlabel(r"Peak logical qubits $Q_{\mathrm{peak}}$")
    ax_b.set_ylabel(r"Expected Toffolis $\hat{T}_{\mathrm{fact}}$ ($10^9$)")
    ax_b.set_title("(b) Vs. concurrent work (Sharma & Lau, 2026)", pad=5, fontsize=8.3)
    ax_b.grid(True, alpha=0.22, lw=0.45)
    ax_b.legend(loc="upper right", fontsize=5.35, frameon=True, facecolor="white")

    fig.tight_layout(pad=0.48, w_pad=1.05)
    fig.savefig(OUT_DIR / "fig5_accounting_and_concurrent.pdf")
    fig.savefig(OUT_DIR / "fig5_accounting_and_concurrent.png", dpi=240)
    plt.close(fig)

    print(f"Reproduced Figs. 1, 3, and 5 (plus supp_convention_compare) in {OUT_DIR}")


if __name__ == "__main__":
    main()
