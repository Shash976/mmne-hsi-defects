# Anomaly-detection report — `sio2_dish_black`

Unsupervised spectral-anomaly screening of SiO₂ thin-film pieces (no defect labels, no reference spectra). Two anomaly products per piece: the **within-film** map (detectors fit on the film's own majority; drives the flagged regions below) and the **silicon-baseline contrast** map (distance from the `sio2_bare_si` control population — the hypothesis deliverable). Regions are *described*, never named as defect types.

## Run configuration

- Target: `sio2_dish_black` · Baseline: `sio2_bare_si`
- PCA components: 3 — explained variance: PC1=84.2%, PC2=9.2%, PC3=2.7%
- Clustering: `kmeans` (k=4)
- Anomaly detectors: ['iforest', 'mahalanobis'] · fit_on=`self` · flag percentile 97.5
- Postprocessing: median=3, opening=1, min_component=25

Sample inventory: [`data\samples.csv`](data/samples.csv)

## Per-piece summary

| piece | silhouette | clusters | anomalous | regions | largest (px) | edge share | median Si-dist |
|---|---|---|---|---|---|---|---|
| Dish on Black - 1_p01 | 0.37 | 4 | 0.00% | 0 | 0 | nan | 607.8 |
| Dish on Black - 1_p02 | 0.53 | 4 | 0.00% | 0 | 0 | nan | 822.7 |
| Dish on Black - 1_p03 | 0.41 | 4 | 3.05% | 2 | 117 | 0% | 641.8 |
| Dish on Black - 1_p04 | 0.33 | 4 | 0.00% | 0 | 0 | nan | 477.6 |
| Dish on Black - 1_p05 | 0.38 | 4 | 0.00% | 0 | 0 | nan | 415.5 |
| Dish on Black - 1_p06 | 0.40 | 4 | 0.00% | 0 | 0 | nan | 396.9 |
| Dish on Black - 1_p07 | 0.37 | 4 | 0.00% | 0 | 0 | nan | 411.3 |
| Dish on Black - 1_p08 | 0.41 | 4 | 0.00% | 0 | 0 | nan | 385.9 |
| Dish on Black - 1_p09 | 0.39 | 4 | 0.00% | 0 | 0 | nan | 578.1 |
| Dish on Black - 1_p10 | 0.43 | 4 | 4.30% | 1 | 58 | 100% | 503.4 |
| Dish on Black - 1_p11 | 0.44 | 4 | 0.00% | 0 | 0 | nan | 526.8 |
| Dish on Black - 1_p12 | 0.45 | 4 | 0.00% | 0 | 0 | nan | 438.3 |
| Dish on Black - 1_p13 | 0.41 | 4 | 0.00% | 0 | 0 | nan | 448.5 |

## The document's questions

- **Localized?** Mean anomalous fraction across pieces is 0.56% (3 region(s) total). Small and localized — consistent with the expected 2–10% band.
- **Repeated across pieces?** 2/13 pieces have at least one flagged region. Recurring regions in similar positions across pieces suggest a process signature; isolated ones suggest local events.
- **Near edges?** On average 50% of flagged pixels lie within 5 px of the piece boundary. Not edge-dominated.
- **Random?** Compare the flagged-region overlay against the cluster map in each `<piece>_analysis.png`: regions that respect cluster boundaries are spectrally coherent populations; scattered speckle that survives postprocessing suggests noise.

## Silicon baseline vs processed film

`median Si-dist` above is each piece's median Mahalanobis distance from the bare-silicon control population (spectral space). Uniformly large values simply reflect the material difference (SiO₂ ≠ Si); *variation* between pieces or within a piece (see the baseline-contrast panel) is the interesting signal.

## Artifacts

- `pca_summary.png` — explained variance + PC loadings
- `pca_scatter.png` — PC1 vs PC2 by piece
- `spectral_histogram.png` — distribution of analysis values
- `<piece>_analysis.png` — 9-panel maps (PCs, clusters, anomaly, baseline contrast, spectral distance, probability, regions, spectra, histogram)
- `<piece>_regions.csv` — region tables (always written; empty = none flagged)
- `roi_table.csv` — cross-specimen ROI ML table
- `cluster_comparison.csv` — method-stability comparison (when requested)
- `roi_evaluation.csv` — specimen-level hold-out scores (when ≥2 specimens)

## Not established here (future work)

Physical origin of any region requires SEM/AFM/Raman/XPS/TEM follow-up (document Stage 12). This report only ranks *where* to look.
