# Anomaly-detection report — `sio2_clean`

Unsupervised spectral-anomaly screening of SiO₂ thin-film pieces (no defect labels, no reference spectra). Two anomaly products per piece: the **within-film** map (detectors fit on the film's own majority; drives the flagged regions below) and the **silicon-baseline contrast** map (distance from the `sio2_bare_si` control population — the hypothesis deliverable). Regions are *described*, never named as defect types.

## Run configuration

- Target: `sio2_clean` · Baseline: `sio2_bare_si`
- PCA components: 3 — explained variance: PC1=66.0%, PC2=19.0%, PC3=4.8%
- Clustering: `kmeans` (k=4)
- Anomaly detectors: ['iforest', 'mahalanobis'] · fit_on=`self` · flag percentile 97.5
- Postprocessing: median=3, opening=1, min_component=25

Sample inventory: [`data\samples.csv`](data/samples.csv)

## Per-piece summary

| piece | silhouette | clusters | anomalous | regions | largest (px) | edge share | median Si-dist |
|---|---|---|---|---|---|---|---|
| Clean 8_7_p01 | 0.37 | 4 | 0.00% | 0 | 0 | nan | 245.4 |
| Clean 8_7_p02 | 0.49 | 4 | 7.38% | 2 | 495 | 11% | 412.6 |
| Clean 8_7_p03 | 0.43 | 4 | 0.00% | 0 | 0 | nan | 195.5 |
| Clean 8_7_p04 | 0.39 | 4 | 0.00% | 0 | 0 | nan | 204.0 |
| Clean 8_7_p05 | 0.38 | 4 | 0.00% | 0 | 0 | nan | 163.5 |

## The document's questions

- **Localized?** Mean anomalous fraction across pieces is 1.48% (2 region(s) total). Small and localized — consistent with the expected 2–10% band.
- **Repeated across pieces?** 1/5 pieces have at least one flagged region. Recurring regions in similar positions across pieces suggest a process signature; isolated ones suggest local events.
- **Near edges?** On average 11% of flagged pixels lie within 5 px of the piece boundary. Not edge-dominated.
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
