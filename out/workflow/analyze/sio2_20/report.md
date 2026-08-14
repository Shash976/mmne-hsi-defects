# Anomaly-detection report — `sio2_20`

Unsupervised spectral-anomaly screening of SiO₂ thin-film pieces (no defect labels, no reference spectra). Two anomaly products per piece: the **within-film** map (detectors fit on the film's own majority; drives the flagged regions below) and the **silicon-baseline contrast** map (distance from the `sio2_bare_si` control population — the hypothesis deliverable). Regions are *described*, never named as defect types.

## Run configuration

- Target: `sio2_20` · Baseline: `sio2_bare_si`
- PCA components: 5 — explained variance: PC1=84.2%, PC2=6.1%, PC3=2.5%, PC4=1.6%, PC5=1.3%
- Clustering: `kmeans` (k=5)
- Anomaly detectors: ['iforest', 'mahalanobis'] · fit_on=`self` · flag percentile 97.5
- Postprocessing: median=3, opening=1, min_component=25

Sample inventory: [`data\samples.csv`](data/samples.csv)

## Per-piece summary

| piece | silhouette | clusters | anomalous | regions | largest (px) | edge share | median Si-dist |
|---|---|---|---|---|---|---|---|
| sio2 all 20 dish white_p01 | 0.40 | 5 | 3.10% | 3 | 516 | 21% | 234.3 |
| sio2 all 20 dish white_p02 | 0.46 | 5 | 0.00% | 0 | 0 | nan | 183.1 |
| sio2 all 20 dish white_p03 | 0.31 | 5 | 1.60% | 2 | 75 | 0% | 301.0 |
| sio2 all 20 dish white_p04 | 0.32 | 5 | 0.00% | 0 | 0 | nan | 259.3 |
| sio2 all 20 dish white_p05 | 0.38 | 5 | 0.74% | 1 | 46 | 0% | 225.9 |
| sio2 all 20 dish white_p06 | 0.46 | 5 | 0.00% | 0 | 0 | nan | 183.3 |
| sio2 all 20 dish white_p07 | 0.30 | 5 | 0.00% | 0 | 0 | nan | 301.9 |
| sio2 all 20 dish white_p08 | 0.38 | 5 | 11.16% | 1 | 558 | 0% | 578.8 |
| sio2 all 20 dish white_p09 | 0.38 | 5 | 0.00% | 0 | 0 | nan | 216.0 |
| sio2 all 20 dish white_p10 | 0.30 | 5 | 0.00% | 0 | 0 | nan | 251.6 |
| sio2 all 20 dish white_p11 | 0.39 | 5 | 0.00% | 0 | 0 | nan | 246.3 |
| sio2 all 20 dish white_p12 | 0.31 | 5 | 0.00% | 0 | 0 | nan | 438.8 |
| sio2 all 20 dish white_p13 | 0.47 | 5 | 5.43% | 1 | 90 | 92% | 204.2 |
| sio2 all 20 dish white_p14 | 0.32 | 5 | 0.00% | 0 | 0 | nan | 249.5 |
| sio2 all 20 dish white_p15 | 0.50 | 5 | 0.00% | 0 | 0 | nan | 229.4 |

## The document's questions

- **Localized?** Mean anomalous fraction across pieces is 1.47% (8 region(s) total). Small and localized — consistent with the expected 2–10% band.
- **Repeated across pieces?** 5/15 pieces have at least one flagged region. Recurring regions in similar positions across pieces suggest a process signature; isolated ones suggest local events.
- **Near edges?** On average 23% of flagged pixels lie within 5 px of the piece boundary. Not edge-dominated.
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
