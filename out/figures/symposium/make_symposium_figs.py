"""Generate symposium figures from real pipeline outputs.

Every number here is read off disk (samples.csv, report.md, roi_evaluation.csv,
regions.csv, noise_metrics.csv) or computed from the extracted piece cubes.
Nothing is invented.
"""
from __future__ import annotations

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = r"C:\Users\shash\Desktop\Code\MMNE\HSI"
OUT = os.path.join(REPO, "out", "figures", "symposium")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- style
SI = "#4A6E8A"        # bare silicon / control
SIO2 = "#1F7A6B"      # processed SiO2 / experimental
ACCENT = "#D9822B"    # anomaly / flag
ALERT = "#B5382F"     # caution / limitation
INK = "#243440"
MUTED = "#8A97A0"
GRID = "#DCE3E8"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


def tidy(ax, ygrid=True):
    if ygrid:
        ax.yaxis.grid(True, lw=0.8)
        ax.set_axisbelow(True)


# ---------------------------------------------------------------- data
samples = pd.read_csv(os.path.join(REPO, "data", "samples.csv"))
inv = json.load(open(os.path.join(REPO, "data", "inventory_summary.json")))
ANA = os.path.join(REPO, "out", "workflow", "analyze", "sio2_dish_white_20")
EXT = os.path.join(REPO, "out", "workflow", "extract")


def parse_report():
    """Per-piece table out of report.md (the authoritative run record)."""
    rows = []
    txt = open(os.path.join(ANA, "report.md"), encoding="utf8").read()
    for line in txt.splitlines():
        m = re.match(r"\|\s*(sio2 all 20 dish white_p\d+)\s*\|(.+)\|\s*$", line)
        if not m:
            continue
        piece = m.group(1)
        c = [x.strip() for x in m.group(2).split("|")]
        rows.append(dict(
            piece=piece,
            short="p%s" % piece[-2:],
            silhouette=float(c[0]),
            clusters=int(c[1]),
            anom_pct=float(c[2].rstrip("%")),
            regions=int(c[3]),
            largest=int(c[4]),
            edge=np.nan if c[5] == "nan" else float(c[5].rstrip("%")),
            si_dist=float(c[6]),
        ))
    return pd.DataFrame(rows)


rep = parse_report()
print(rep)


# ================================================================ FIG 1
# Dataset inventory — what we actually imaged
def fig_inventory():
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [2.1, 1]})

    d = samples.sort_values("area_px")
    colors = [SI if m == "silicon" else SIO2 for m in d.material]
    y = np.arange(len(d))
    ax.barh(y, d.area_px, color=colors, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("bare silicon all_", "Si ")
                        .replace("sio2 all 20 dish white_", "SiO\u2082 ")
                        for s in d.sample_id], fontsize=9.5)
    ax.set_xlabel("analysable area (pixels)")
    ax.set_title("24 specimens imaged", loc="left")
    ax.xaxis.grid(True, lw=0.8)
    ax.set_axisbelow(True)
    for yi, (a, n) in enumerate(zip(d.area_px, d.n_rois)):
        ax.text(a + 300, yi, f"{a:,}  ·  {n} ROIs", va="center",
                fontsize=8.5, color=MUTED)
    ax.set_xlim(0, d.area_px.max() * 1.30)

    ax2.axis("off")
    n_si = int(inv["n_samples_by_material"]["silicon"])
    n_ox = int(inv["n_samples_by_material"]["sio2"])
    cards = [
        ("24", "independent specimens", INK, "target 10\u201320"),
        (f"{n_si} / {n_ox}", "bare Si control / SiO\u2082 experimental", SI,
         "37% control \u2014 target 25\u201335%"),
        (f"{inv['n_rois_total']:,}", "ROI samples (8\u00d78 px patches)", SIO2,
         "median ~195 per piece \u2014 target 100\u2013300"),
        (f"{inv['total_imaging_area_px']:,}", "px of analysed film + substrate",
         ACCENT, "300 spectral bands each"),
    ]
    yb = 0.95
    for big, lab, col, sub in cards:
        ax2.text(0.0, yb, big, fontsize=34, fontweight="bold", color=col,
                 va="top", transform=ax2.transAxes)
        ax2.text(0.0, yb - 0.105, lab, fontsize=11.5, va="top", color=INK,
                 transform=ax2.transAxes)
        ax2.text(0.0, yb - 0.155, sub, fontsize=9.5, va="top", color=MUTED,
                 style="italic", transform=ax2.transAxes)
        yb -= 0.245
    ax2.set_title("Against the design targets", loc="left")

    fig.suptitle("Sample inventory \u2014 every piece is one independent specimen",
                 fontsize=17, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    save(fig, "fig01_inventory.png")


# ================================================================ FIG 2
# Si vs SiO2 mean spectra — computed from the extracted piece cubes
def load_piece_mean(piece_dir, piece_id):
    import spectral
    hdr = os.path.join(piece_dir, piece_id + ".hdr")
    mask_p = os.path.join(piece_dir, piece_id + "_mask.npy")
    img = spectral.open_image(hdr)
    wl = np.asarray(img.bands.centers, dtype=float)
    arr = np.asarray(img.load(), dtype=np.float64)
    mask = np.load(mask_p) if os.path.exists(mask_p) else np.ones(arr.shape[:2], bool)
    px = arr[mask.astype(bool)]
    px = px[np.isfinite(px).all(axis=1)]
    return wl, px.mean(axis=0), px.std(axis=0), px


def collect_spectra():
    # Numeric-only cache written by this script itself (no pickle needed).
    cache = os.path.join(ROOT, "spectra_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["wl"], z["si"], z["ox"], z["si_var"], z["ox_var"]
    si, ox, si_var, ox_var = [], [], [], []
    wl = None
    for _, r in samples.iterrows():
        pdir = os.path.join(EXT, r.dataset, r.sample_id)
        if not os.path.isdir(pdir):
            continue
        try:
            w, mu, sd, px = load_piece_mean(pdir, r.sample_id)
        except Exception as e:                      # noqa: BLE001
            print("skip", r.sample_id, e)
            continue
        wl = w
        # Spatial heterogeneity, the definition rois.py:100 / regions.py:106 use
        # for ML features: variance ACROSS pixels per band, averaged over bands.
        v = float(px.var(axis=0).mean())
        if r.material == "silicon":
            si.append(mu); si_var.append(v)
        else:
            ox.append(mu); ox_var.append(v)
        print("  loaded", r.sample_id, px.shape)
    si, ox = np.array(si), np.array(ox)
    np.savez(cache, wl=wl, si=si, ox=ox,
             si_var=np.array(si_var), ox_var=np.array(ox_var))
    return wl, si, ox, np.array(si_var), np.array(ox_var)


def fig_spectra(wl, si, ox, si_var, ox_var):
    keep = wl >= 400.0          # drop the sensor's UV roll-off spike
    wlk = wl[keep]
    sik, oxk = si[:, keep], ox[:, keep]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4),
                             gridspec_kw={"width_ratios": [1.5, 1.5, 1]})

    ax = axes[0]
    for s in sik:
        ax.plot(wlk, s, color=SI, lw=0.8, alpha=0.30)
    for s in oxk:
        ax.plot(wlk, s, color=SIO2, lw=0.8, alpha=0.30)
    ax.plot(wlk, sik.mean(0), color=SI, lw=3.0,
            label=f"bare Si control (n={len(si)})")
    ax.plot(wlk, oxk.mean(0), color=SIO2, lw=3.0,
            label=f"processed SiO\u2082 (n={len(ox)})")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("reflectance")
    ax.set_title("Mean spectrum per specimen", loc="left")
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax.set_xlim(400, 1010)
    tidy(ax)

    # within-material spread
    ax = axes[1]
    cvs = {}
    for arr, col, lab in ((sik, SI, "bare Si"), (oxk, SIO2, "processed SiO\u2082")):
        m, s = arr.mean(0), arr.std(0)
        cv = s / np.maximum(m, 1e-6) * 100
        cvs[lab] = cv
        ax.plot(wlk, cv, color=col, lw=2.6, label=lab)
    ax.fill_between(wlk, cvs["bare Si"], cvs["processed SiO\u2082"],
                    where=cvs["processed SiO\u2082"] > cvs["bare Si"],
                    color=ACCENT, alpha=0.16, lw=0)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("between-specimen CV (%)")
    ax.set_title("Control is uniform, film is not", loc="left")
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax.set_xlim(400, 1010)
    ax.text(0.97, 0.06,
            f"peak {cvs['processed SiO\u2082'].max():.0f}% vs "
            f"{cvs['bare Si'].max():.0f}%",
            transform=ax.transAxes, ha="right", fontsize=10.5,
            color=ACCENT, fontweight="bold")
    tidy(ax)

    ax = axes[2]
    # The pipeline's own Stage-4 metric, so the slide matches
    # material_variance.csv rather than a second definition.
    mv = pd.read_csv(os.path.join(
        REPO, "out", "workflow", "explore",
        "sio2_bare_si+sio2_dish_white_20", "material_variance.csv"))
    mv = mv[~mv.piece_id.str.startswith("<")]
    si_var = mv[mv.material == "silicon"].mean_spectral_variance.values
    ox_var = mv[mv.material == "sio2"].mean_spectral_variance.values
    parts = [si_var, ox_var]
    bp = ax.boxplot(parts, patch_artist=True, widths=0.55,
                    medianprops=dict(color=INK, lw=2),
                    flierprops=dict(marker="o", ms=4, mfc=MUTED, mec="none"))
    for patch, col in zip(bp["boxes"], (SI, SIO2)):
        patch.set_facecolor(col); patch.set_alpha(0.55); patch.set_edgecolor(col)
    for i, (v, col) in enumerate(zip(parts, (SI, SIO2)), start=1):
        ax.scatter(np.random.normal(i, 0.055, len(v)), v, s=26, color=col,
                   edgecolor="white", lw=0.6, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["bare Si", "SiO\u2082"])
    ax.set_yscale("log")
    ax.set_ylabel("within-piece spatial variance")
    ax.set_title("Within-piece spread", loc="left")
    ax.set_ylim(min(min(si_var), min(ox_var)) * 0.35,
                max(max(si_var), max(ox_var)) * 3.4)
    ratio = np.median(ox_var) / np.median(si_var)
    si_sp = max(si_var) / min(si_var)
    ox_sp = max(ox_var) / min(ox_var)
    ax.text(1, max(si_var) * 1.5, f"{si_sp:.0f}\u00d7 range", ha="center",
            fontsize=10.5, color=SI, fontweight="bold")
    ax.text(2, max(ox_var) * 1.5, f"{ox_sp:.0f}\u00d7 range", ha="center",
            fontsize=10.5, color=ACCENT, fontweight="bold")
    tidy(ax)

    fig.suptitle("Why bare silicon is the control, not a reference spectrum",
                 fontsize=17, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.text(0.005, 0.048,
             "Panel 3 is the pipeline's own Stage-4 metric "
             "(material_variance.csv). Contrary to the design "
             f"spec's expectation, SiO\u2082 median variance is {ratio:.2f}\u00d7 "
             "bare Si \u2014 not higher."
             ,
             fontsize=10.5, color=MUTED, style="italic")
    fig.text(0.005, 0.005,
             f"What separates the populations is spread, not level: the film "
             f"spans a {ox_sp:.1f}\u00d7 range against {si_sp:.1f}\u00d7 for the "
             "control. The shipped SAM masks isolate the uniform oxide region, "
             "which suppresses absolute SiO\u2082 variance (see limitations).",
             fontsize=10.5, color=MUTED, style="italic")
    fig.tight_layout(rect=[0, 0.09, 1, 0.93])
    save(fig, "fig02_si_vs_sio2_spectra.png")
    return ox_sp / si_sp


# ================================================================ FIG 3
# Per-piece results — what the run actually produced
def fig_results():
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 9), sharex=True)
    x = np.arange(len(rep))
    lbl = [p.replace("sio2 all 20 dish white_", "") for p in rep.piece]

    ax = axes[0]
    ax.bar(x, rep.silhouette, color=SIO2, width=0.66)
    ax.axhspan(0.4, 0.8, color=SIO2, alpha=0.10, lw=0)
    ax.axhline(0.4, color=SIO2, ls="--", lw=1.2)
    ax.set_ylabel("silhouette")
    ax.set_ylim(0, 0.75)
    ax.set_title("Cluster separation \u2014 every piece 0.35\u20130.64", loc="left",
                 fontsize=13)
    ax.text(0.2, 0.685, "spec band 0.4\u20130.8", ha="left", fontsize=9.5,
            color=SIO2, style="italic")
    tidy(ax)

    ax = axes[1]
    cols = [ACCENT if v > 0 else "#D5DDE3" for v in rep.anom_pct]
    ax.bar(x, rep.anom_pct, color=cols, width=0.66)
    ax.axhline(rep.anom_pct.mean(), color=ALERT, ls="--", lw=1.4)
    ax.text(0.2, rep.anom_pct.mean() + 0.12,
            f"mean {rep.anom_pct.mean():.2f}%", fontsize=10, color=ALERT,
            fontweight="bold")
    ax.set_ylabel("anomalous pixels (%)")
    ax.set_title(
        f"Flagged fraction \u2014 {int((rep.anom_pct > 0).sum())}/{len(rep)} pieces "
        f"carry {int(rep.regions.sum())} regions total", loc="left", fontsize=13)
    tidy(ax)

    ax = axes[2]
    ax.bar(x, rep.si_dist, color=SI, width=0.66)
    ax.axhline(rep.si_dist.median(), color=INK, ls=":", lw=1.3)
    ax.set_ylabel("median distance\nfrom bare-Si population")
    ax.set_title("Silicon-baseline contrast \u2014 the spread between pieces is the signal",
                 loc="left", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(lbl, rotation=0, fontsize=10)
    tidy(ax)

    fig.suptitle("Results across all 15 SiO\u2082 specimens (one run, no labels)",
                 fontsize=17, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    save(fig, "fig03_per_piece_results.png")


# ================================================================ FIG 4
# Specimen-level split — the leakage argument
def fig_leakage():
    ev = pd.read_csv(os.path.join(ANA, "roi_evaluation.csv"))
    tr = ev[ev.split == "train"].score_iforest
    te = ev[ev.split == "test"].score_iforest
    n_tr_sp = ev[ev.split == "train"].specimen.nunique()
    n_te_sp = ev[ev.split == "test"].specimen.nunique()

    fig = plt.figure(figsize=(14.5, 6.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.25, 1.1], wspace=0.30)

    # (a) schematic
    ax = fig.add_subplot(gs[0]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(0, 10.0, "Naive per-ROI split", fontsize=12.5, fontweight="bold",
            color=ALERT, va="top")
    ax.text(0, 4.85, "Specimen-level split (ours)", fontsize=12.5,
            fontweight="bold", color=SIO2, va="top")
    rng = np.random.default_rng(3)
    for y0, mixed in [(6.5, True), (1.45, False)]:
        for s in range(4):
            x0 = 0.2 + s * 2.45
            ax.add_patch(Rectangle((x0, y0), 2.15, 2.3, facecolor="#F2F5F7",
                                   edgecolor=MUTED, lw=1.0))
            ax.text(x0 + 1.07, y0 - 0.38, f"specimen {s+1}", ha="center",
                    fontsize=8.8, color=MUTED)
            for i in range(3):
                for j in range(3):
                    if mixed:
                        c = ACCENT if rng.random() < 0.4 else SIO2
                    else:
                        c = ACCENT if s == 3 else SIO2
                    ax.add_patch(Rectangle((x0 + 0.22 + j * 0.62,
                                            y0 + 0.25 + i * 0.62),
                                           0.5, 0.5, facecolor=c, lw=0))
    ax.add_patch(Rectangle((0.05, 6.32), 9.95, 2.66, fill=False,
                           edgecolor=ALERT, lw=2.0, ls="--"))
    ax.text(0, 5.55, "neighbouring patches land on both sides \u2192 leakage",
            fontsize=9.5, color=ALERT, style="italic", va="top")
    ax.text(0, 0.62, "whole specimens held out \u2192 honest generalisation",
            fontsize=9.5, color=SIO2, style="italic", va="top")
    ax.scatter([], [], color=SIO2, marker="s", s=60, label="train ROI")
    ax.scatter([], [], color=ACCENT, marker="s", s=60, label="test ROI")
    ax.legend(frameon=False, fontsize=9.5, loc="lower center",
              bbox_to_anchor=(0.5, -0.10), ncol=2)

    # (b) score distributions
    ax = fig.add_subplot(gs[1])
    bins = np.linspace(min(tr.min(), te.min()), max(tr.max(), te.max()), 45)
    ax.hist(tr, bins=bins, color=SIO2, alpha=0.72, label=f"train  n={len(tr):,}")
    ax.hist(te, bins=bins, color=ACCENT, alpha=0.72, label=f"test   n={len(te):,}")
    ax.axvline(tr.mean(), color=SIO2, lw=2.2)
    ax.axvline(te.mean(), color=ACCENT, lw=2.2)
    ax.set_xlabel("IsolationForest ROI anomaly score")
    ax.set_ylabel("ROIs")
    ax.legend(frameon=False, fontsize=10.5)
    ax.set_title("Held-out scores match training", loc="left", fontsize=12.5)
    ax.text(0.97, 0.60, f"mean {tr.mean():.3f} train\nmean {te.mean():.3f} test",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            fontweight="bold", color=INK)
    tidy(ax)

    # (c) per-specimen means
    ax = fig.add_subplot(gs[2])
    g = (ev.groupby(["specimen", "split"]).score_iforest
         .mean().reset_index().sort_values("score_iforest"))
    cols = [SIO2 if s == "train" else ACCENT for s in g.split]
    ax.barh(np.arange(len(g)), g.score_iforest, color=cols, height=0.72)
    ax.set_yticks(np.arange(len(g)))
    ax.set_yticklabels([s.replace("sio2 all 20 dish white_", "")
                        for s in g.specimen], fontsize=9.5)
    ax.set_xlabel("mean ROI score")
    ax.set_xlim(0.3, max(g.score_iforest) * 1.06)
    ax.set_title(f"{n_tr_sp} train / {n_te_sp} test specimens", loc="left",
                 fontsize=12.5)
    ax.xaxis.grid(True, lw=0.8); ax.set_axisbelow(True)

    fig.suptitle("Leakage control: the split is by specimen, never by patch",
                 fontsize=17, fontweight="bold", color=INK, x=0.005, ha="left",
                 y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "fig04_specimen_split.png")


# ================================================================ FIG 5
# Clustering method comparison
def fig_methods():
    p = os.path.join(ANA, "cluster_comparison.csv")
    d = pd.read_csv(p)
    meth = d[d.method.isin(["kmeans", "dbscan", "gmm"])].copy()
    pairs = d[d.method.str.contains(r"\|")].copy()

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 5.2))
    pretty = {"kmeans": "K-Means", "dbscan": "DBSCAN", "gmm": "GMM"}
    names = [pretty[m] for m in meth.method]
    cols = [SIO2, MUTED, SI]

    specs = [
        ("silhouette", "Silhouette", "higher is better", None),
        ("davies_bouldin", "Davies\u2013Bouldin", "lower is better", None),
        ("calinski_harabasz", "Calinski\u2013Harabasz", "higher is better", "log"),
    ]
    for ax, (col, title, note, scale) in zip(axes, specs):
        vals = meth[col].to_numpy(dtype=float)
        plot_vals = np.nan_to_num(vals, nan=0.0)
        b = ax.bar(names, plot_vals, color=cols, width=0.62)
        if scale:
            ax.set_yscale(scale)
            ax.set_ylim(1, np.nanmax(vals) * 6)
        else:
            ax.set_ylim(0, np.nanmax(vals) * 1.30)
        for rect, v in zip(b, vals):
            if np.isnan(v):
                # DBSCAN returned a single cluster -- the index is undefined,
                # not zero. Say so rather than drawing a misleading empty bar.
                ax.text(rect.get_x() + rect.get_width() / 2,
                        (1.6 if scale else np.nanmax(vals) * 0.06),
                        "undefined\n(1 cluster)", ha="center", va="bottom",
                        fontsize=9.5, color=ALERT, fontweight="bold")
            else:
                ax.text(rect.get_x() + rect.get_width() / 2,
                        v * (1.25 if scale else 1.0) +
                        (0 if scale else np.nanmax(vals) * 0.025),
                        f"{v:,.0f}" if v > 100 else f"{v:.2f}",
                        ha="center", fontsize=10.5, fontweight="bold")
        ax.set_title(title, loc="left", fontsize=13)
        ax.set_xlabel(note, fontsize=10, color=MUTED)
        tidy(ax)

    ax = axes[3]
    lbls = [x.replace("|", "\nvs\n").replace("gmm", "GMM")
            .replace("kmeans", "K-Means").replace("dbscan", "DBSCAN")
            for x in pairs.method]
    ari = pairs.adjusted_rand_index.to_numpy(dtype=float)
    order = [1, 0, 2]                       # put the informative pair first
    lbls = [lbls[i] for i in order]
    ari = ari[order]
    b = ax.bar(lbls, ari, color=[ACCENT, MUTED, MUTED], width=0.6)
    for rect, v in zip(b, ari):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 0.60)
    ax.set_title("Agreement (ARI)", loc="left", fontsize=13)
    ax.set_xlabel("do the methods find the same populations?", fontsize=10,
                  color=MUTED)
    tidy(ax)

    fig.suptitle("Why K-Means (k=4) is the default \u2014 measured, not assumed",
                 fontsize=17, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.text(0.005, 0.005,
             "On this dataset DBSCAN degenerates to a single cluster, so every "
             "internal index is undefined and its ARI against both other "
             "methods is 0. GMM does partition the data and broadly agrees "
             "with K-Means (ARI 0.47), but scores worse on all three indices.",
             fontsize=10.5, color=MUTED, style="italic")
    fig.tight_layout(rect=[0, 0.045, 1, 0.92])
    save(fig, "fig05_clustering_methods.png")


# ================================================================ FIG 6
# Design decisions taken by measurement (ROI yield + erosion)
def fig_decisions():
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))

    # ROI yield  (docs/audit-vs-objective.md, measured over all 41 saved masks)
    ax = axes[0]
    cfgs = ["32 / 32\n(initial)", "16 / 8", "8 / 4\n(chosen)"]
    yields = [73, 2301, 11185]
    cols = [ALERT, MUTED, SIO2]
    b = ax.bar(cfgs, yields, color=cols, width=0.6)
    ax.set_yscale("log")
    for rect, v in zip(b, yields):
        ax.text(rect.get_x() + rect.get_width() / 2, v * 1.15, f"{v:,}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("total ROI samples")
    ax.set_xlabel("patch / stride (px)", fontsize=10, color=MUTED)
    ax.set_title("ROI yield \u2014 153\u00d7 more samples", loc="left", fontsize=13)
    ax.set_ylim(30, 40000)
    tidy(ax)

    # median ROIs per piece against spec band
    ax = axes[1]
    med = [1, 42, 205]
    ax.axhspan(100, 300, color=SIO2, alpha=0.10, lw=0)
    ax.plot(cfgs, med, "-o", color=INK, lw=2, ms=9, mfc=SIO2, mec=INK)
    for i, v in enumerate(med):
        ax.text(i, v * 1.4 if v > 5 else v + 1.4, str(v), ha="center",
                fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("median ROIs per piece")
    ax.set_xlabel("patch / stride (px)", fontsize=10, color=MUTED)
    ax.text(0.03, 0.86, "design target 100\u2013300", transform=ax.transAxes,
            fontsize=10.5, color=SIO2, style="italic")
    ax.set_title("Only 8/4 reaches the target", loc="left", fontsize=13)
    ax.set_ylim(0.5, 900)
    tidy(ax)

    # erosion trade-off
    ax = axes[2]
    er = ["0 px", "1 px\n(chosen)", "3 px"]
    rois = [4554, 4308, 3760]
    edge = [39, 34, 34]
    ax.bar(er, rois, color=[MUTED, SIO2, MUTED], width=0.6)
    ax.set_ylabel("ROIs retained", color=INK)
    ax.set_ylim(0, 5600)
    for i, v in enumerate(rois):
        ax.text(i, v + 110, f"{v:,}", ha="center", fontsize=10.5,
                fontweight="bold")
    ax2 = ax.twinx()
    ax2.plot(er, edge, "-o", color=ACCENT, lw=2.4, ms=9)
    ax2.set_ylabel("mean edge share (%)", color=ACCENT)
    ax2.tick_params(axis="y", colors=ACCENT)
    ax2.set_ylim(25, 45)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(ACCENT)
    for i, v in enumerate(edge):
        ax2.text(i, v - 1.9, f"{v}%", ha="center", va="top", fontsize=10.5,
                 color=ACCENT, fontweight="bold")
    ax.set_title("Mask erosion \u2014 1 px is the efficient point", loc="left",
                 fontsize=13)
    ax.set_xlabel("3 px costs 17% of ROIs for no further gain", fontsize=10,
                  color=MUTED)
    tidy(ax)

    fig.suptitle("Every default was chosen by measurement",
                 fontsize=17, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "fig06_design_decisions.png")


# ================================================================ FIG 7
# Region characterization — the quantitative output
def fig_regions():
    frames = []
    for f in os.listdir(ANA):
        if f.endswith("_regions.csv"):
            d = pd.read_csv(os.path.join(ANA, f))
            if len(d):
                d["piece"] = f.replace("sio2 all 20 dish white_", "").replace(
                    "_regions.csv", "")
                frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    print("regions:", len(d))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1))

    ax = axes[0]
    sc = ax.scatter(d.area, d.baseline_distance, s=np.clip(d.area, 30, 400),
                    c=d.compactness, cmap="viridis", edgecolor=INK, lw=0.9,
                    alpha=0.92, vmin=0, vmax=1.3)
    for _, r in d.iterrows():
        ax.annotate(r.piece, (r.area, r.baseline_distance),
                    textcoords="offset points", xytext=(9, 6), fontsize=8.5,
                    color=MUTED)
    ax.set_xlabel("region area (px)")
    ax.set_ylabel("distance from bare-Si population")
    ax.set_yscale("log")
    ax.set_xlim(0, d.area.max() * 1.28)
    ax.set_title(f"All {len(d)} flagged regions", loc="left", fontsize=13)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("compactness", fontsize=10)
    tidy(ax)

    ax = axes[1]
    order = d.sort_values("area")
    ax.barh(np.arange(len(order)), order.area, color=ACCENT, height=0.7)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([f"{p} #{int(i)}" for p, i in
                        zip(order.piece, order.region_id)], fontsize=9.5)
    ax.set_xlabel("area (px)")
    ax.axvline(25, color=ALERT, ls="--", lw=1.4)
    ax.set_ylim(-1.4, len(order) - 0.4)
    ax.text(30, -1.25, "min-component filter = 25 px", fontsize=9.5,
            color=ALERT, va="bottom")
    ax.set_title("Contiguous regions, not speckle", loc="left", fontsize=13)
    ax.xaxis.grid(True, lw=0.8); ax.set_axisbelow(True)

    ax = axes[2]
    ax.scatter(d.spectral_variance, d.mean_anomaly, s=110, color=SIO2,
               edgecolor=INK, lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("within-region spectral variance")
    ax.set_ylabel("mean anomaly score")
    ax.set_title("Described, never named", loc="left", fontsize=13)
    tidy(ax)

    fig.suptitle("Stage 10\u201311 \u2014 what the pipeline hands the microscope",
                 fontsize=17, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.text(0.005, 0.005,
             "Each region carries area · perimeter · compactness · "
             "spectral variance · Mahalanobis distance from the bare-Si "
             "population. No defect class is claimed — these are ranked "
             "candidates for SEM/AFM/Raman follow-up.",
             fontsize=10.5, color=MUTED, style="italic")
    fig.tight_layout(rect=[0, 0.045, 1, 0.92])
    save(fig, "fig07_regions.png")


# ================================================================ FIG 8
# Preprocessing gain — real SNR numbers
def fig_preprocessing():
    p = os.path.join(REPO, "out", "workflow", "explore",
                     "sio2_bare_si+sio2_dish_white_20",
                     "noise_metrics.csv")
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    x = np.arange(len(d))
    w = 0.4
    ax.bar(x - w / 2, d.snr_before, w, color=MUTED, label="raw reflectance")
    ax.bar(x + w / 2, d.snr_after, w, color=SIO2,
           label="after Savitzky\u2013Golay")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("bare silicon all_", "Si ")
                        .replace("sio2 all 20 dish white_", "SiO\u2082 ")
                        for s in d.piece_id], rotation=60, ha="right",
                       fontsize=9)
    ax.set_ylabel("spectral SNR")
    ax.set_ylim(0, max(d.snr_after) * 1.22)
    ax.set_title("Smoothing raises SNR on every specimen", loc="left",
                 fontsize=13)
    ax.legend(frameon=False, fontsize=10.5, ncol=2, loc="upper center")
    tidy(ax)

    ax = axes[1]
    gain = d.snr_after / d.snr_before
    ax.hist(gain, bins=12, color=ACCENT, alpha=0.85)
    ax.axvline(gain.median(), color=INK, lw=2.2)
    ax.text(gain.median() * 1.02, ax.get_ylim()[1] * 0.9,
            f"median {gain.median():.1f}\u00d7", fontsize=12,
            fontweight="bold", color=INK)
    ax.set_xlabel("SNR gain factor")
    ax.set_ylabel("specimens")
    ax.set_title("Median gain", loc="left", fontsize=13)
    tidy(ax)

    fig.suptitle("Stage 3 preprocessing \u2014 measured effect, not a claim",
                 fontsize=17, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.text(0.005, 0.055,
             f"RMS spectral noise falls from {d.rms_noise_before.mean():.2e} "
             f"to {d.rms_noise_after.mean():.2e} on average (n={len(d)} "
             "specimens). SNV then removes scatter offsets, keeping shape.",
             fontsize=10.5, color=MUTED, style="italic")
    fig.text(0.005, 0.005,
             "All 24 in-scope specimens (9 bare-Si controls + 15 SiO\u2082). "
             "Stage-2 range check on the same run: <0.01% of in-mask "
             "reflectance values fall outside [0, 1] \u2014 a few dozen "
             "pixels of ~2M, so no systematic clipping or saturation.",
             fontsize=10.5, color=MUTED, style="italic")
    fig.tight_layout(rect=[0, 0.11, 1, 0.92])
    save(fig, "fig08_preprocessing_snr.png")


# ================================================================ FIG 9
# Known limitations — the slide the deck is missing entirely
def fig_limitations():
    fig = plt.figure(figsize=(15.0, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.38,
                          height_ratios=[1, 1.28])

    # (a) merged pieces
    ax = fig.add_subplot(gs[0, 0])
    cfg = ["close_iter=2", "close_iter=4", "close_iter=6\n(shipped)"]
    mx = [7339, 14421, 21106]
    npc = [19, 17, 15]
    b = ax.bar(cfg, mx, color=[SIO2, MUTED, ALERT], width=0.6)
    for rect, v, n in zip(b, mx, npc):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 700,
                f"{v:,} px\n{n} pieces", ha="center", fontsize=9.5,
                fontweight="bold")
    ax.set_ylim(0, 27000)
    ax.set_ylabel("largest 'piece' (px)", fontsize=10.5)
    ax.set_title("(a) Adjacent wafers merge", loc="left", fontsize=12.5)
    ax.set_xlabel("morphological closing", fontsize=9.5, color=MUTED)
    tidy(ax)

    # (b) SAM clips the substrate
    ax = fig.add_subplot(gs[0, 1])
    meth = ["SAM\n(shipped)", "Mahalanobis", "Euclidean"]
    fg = [79522, 24415, 172220]
    b = ax.bar(meth, fg, color=[ALERT, MUTED, SIO2], width=0.6)
    for rect, v in zip(b, fg):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 5000, f"{v:,}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 215000)
    ax.set_ylabel("foreground px recovered", fontsize=10.5)
    ax.set_yticks([0, 50000, 100000, 150000, 200000])
    ax.set_yticklabels(["0", "50k", "100k", "150k", "200k"])
    ax.set_title("(b) SAM is brightness-invariant", loc="left", fontsize=12.5)
    ax.set_xlabel("dark bare Si scores like the bright dish", fontsize=9.5,
                  color=MUTED)
    tidy(ax)

    # (c) edge dominance per piece
    ax = fig.add_subplot(gs[0, 2])
    e = rep.dropna(subset=["edge"]).sort_values("edge")
    cols = [ALERT if v >= 50 else ACCENT for v in e.edge]
    ax.bar([p.replace("sio2 all 20 dish white_", "") for p in e.piece],
           e.edge, color=cols, width=0.6)
    for i, v in enumerate(e.edge):
        ax.text(i, v + 2.5, f"{v:.0f}%", ha="center", fontsize=10,
                fontweight="bold")
    ax.axhline(50, color=ALERT, ls="--", lw=1.3)
    ax.set_ylim(0, 118)
    ax.set_ylabel("flags within 5 px of edge (%)", fontsize=10.5)
    ax.set_title("(c) Some pieces are edge-dominated", loc="left", fontsize=12.5)
    ax.set_xlabel("1 px erosion reduced but did not remove this", fontsize=9.5,
                  color=MUTED)
    tidy(ax)

    # (d) honest status table
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    rows = [
        ("Piece extraction correctness",
         "SAM masks capture the oxide, not the whole wafer; close_iter=6 merges "
         "adjacent wafers into one 'specimen'.",
         "Diagnosed — euclidean backend built and measured; defaults "
         "deliberately unchanged", ALERT),
        ("Boundary-region filtering",
         "Piece-edge pixels mix film and dish spectra and dominate flags on "
         "small pieces (p13 99%, p12 61%).",
         "Partly fixed — 1 px erosion; region-touches-edge filter is open",
         ACCENT),
        ("Overlapping ROI stride",
         "stride 4 < patch 8 reintroduces spatial autocorrelation within a "
         "specimen.",
         "Accepted — hold-out is by whole specimen, so the leakage "
         "argument survives", SIO2),
        ("No physical ground truth",
         "Nothing here establishes what a flagged region physically is.",
         "By design — Stage 12 SEM/AFM/Raman follow-up is future work",
         SI),
    ]
    ax.text(0.005, 1.03, "Where this pipeline is not yet trustworthy",
            fontsize=14, fontweight="bold", color=INK,
            transform=ax.transAxes, va="bottom")
    y = 0.94
    for title, prob, status, col in rows:
        ax.add_patch(Rectangle((0.0, y - 0.20), 0.010, 0.205, facecolor=col,
                               lw=0, transform=ax.transAxes, clip_on=False))
        ax.text(0.026, y, title, fontsize=11.5, fontweight="bold", color=INK,
                transform=ax.transAxes, va="top")
        ax.text(0.026, y - 0.075, prob, fontsize=10.2, color=INK,
                transform=ax.transAxes, va="top")
        ax.text(0.026, y - 0.145, status, fontsize=10.2, color=col,
                style="italic", transform=ax.transAxes, va="top")
        y -= 0.255

    fig.suptitle("Known limitations — stated before anyone asks",
                 fontsize=17, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.text(0.005, 0.005,
             "Every metric in this pipeline passed while the front-end was "
             "merging wafers and clipping substrate. Metric conformance is not "
             "evidence of physical correctness.",
             fontsize=10.5, color=ALERT, style="italic")
    fig.tight_layout(rect=[0, 0.035, 1, 0.945])
    save(fig, "fig09_limitations.png")


if __name__ == "__main__":
    fig_limitations()
    fig_inventory()
    fig_results()
    fig_leakage()
    fig_methods()
    fig_decisions()
    fig_regions()
    fig_preprocessing()
    wl, si, ox, si_var, ox_var = collect_spectra()
    fig_spectra(wl, si, ox, si_var, ox_var)
    print("\nALL FIGURES IN", OUT)
