# Legacy scripts

Superseded by the `hsi_workflow/` package. Kept for reference: this is the
earlier LIG (laser-induced graphene) analysis — RX (Mahalanobis) anomaly
detection and PCA + K-means composition classification — together with its
written-up findings in `PIPELINE.md`.

These are **not** part of the new document-driven semiconductor thin-film
workflow and are not imported by it. They still run as a self-contained group
(`composition_pipeline.py` imports `unsupervised_defect.py` from this same
folder), but only against the LIG reflectance dataset they were built for.

Files:

- `unsupervised_defect.py` — RX anomaly detection.
- `composition_pipeline.py` — PCA/K-means + linear spectral unmixing.
- `pipeline.py` — sketch of a supervised (PLS-DA/RF/SVM) next stage.
- `hsi_explore.py` — ad-hoc cube exploration helpers.
- `lig_pipeline.ipynb` — exploratory notebook.
- `PIPELINE.md` — design notes / findings for the above.
