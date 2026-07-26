"""Verify every deliverable docs/usage.md promises exists for the live pair."""
import glob
import os

REPO = r"C:\Users\shash\Desktop\Code\MMNE\HSI"
EXP = os.path.join(REPO, "out", "workflow", "explore",
                   "sio2_bare_si+sio2_dish_white_20")
ANA = os.path.join(REPO, "out", "workflow", "analyze", "sio2_dish_white_20")

single = [
    ("run_organize", "data/samples.csv", os.path.join(REPO, "data", "samples.csv")),
    ("run_organize", "data/inventory_summary.json",
     os.path.join(REPO, "data", "inventory_summary.json")),
    ("run_organize", "data/manifest.json",
     os.path.join(REPO, "data", "manifest.json")),
    ("run_explore", "material_mean_spectra.png",
     os.path.join(EXP, "material_mean_spectra.png")),
    ("run_explore", "material_variance.csv",
     os.path.join(EXP, "material_variance.csv")),
    ("run_explore", "noise_metrics.csv", os.path.join(EXP, "noise_metrics.csv")),
    ("run_explore", "reflectance_histogram.png",
     os.path.join(EXP, "reflectance_histogram.png")),
    ("run_analyze", "pca_summary.png", os.path.join(ANA, "pca_summary.png")),
    ("run_analyze", "pca_scatter.png", os.path.join(ANA, "pca_scatter.png")),
    ("run_analyze", "spectral_histogram.png",
     os.path.join(ANA, "spectral_histogram.png")),
    ("run_analyze", "roi_table.csv", os.path.join(ANA, "roi_table.csv")),
    ("run_analyze", "roi_evaluation.csv", os.path.join(ANA, "roi_evaluation.csv")),
    ("run_analyze", "cluster_comparison.csv",
     os.path.join(ANA, "cluster_comparison.csv")),
    ("run_analyze", "report.md", os.path.join(ANA, "report.md")),
]

globbed = [
    ("run_explore", "<piece>_explore.png", os.path.join(EXP, "*_explore.png"), 24),
    ("run_analyze", "<piece>_analysis.png", os.path.join(ANA, "*_analysis.png"), 15),
    ("run_analyze", "<piece>_regions.csv", os.path.join(ANA, "*_regions.csv"), 15),
]

bad = 0
print(f"{'stage':<13} {'deliverable':<28} status")
print("-" * 72)
for stage, name, path in single:
    ok = os.path.exists(path)
    size = os.path.getsize(path) if ok else 0
    print(f"{stage:<13} {name:<28} {'OK' if ok else 'MISSING':<8} {size:>9,} B")
    bad += not ok
for stage, name, pat, want in globbed:
    n = len(glob.glob(pat))
    ok = n == want
    print(f"{stage:<13} {name:<28} {'OK' if ok else 'MISMATCH':<8} "
          f"{n} files (expected {want})")
    bad += not ok

print("-" * 72)
print("ALL DELIVERABLES PRESENT" if not bad else f"{bad} PROBLEM(S)")
