# Legacy outputs — predate the revised research objective

Everything in this folder was produced by the **pre-revision** scripts in
`../../legacy/` (composition mapping / defect classification for LIG scans) and
does **not** reflect the current unsupervised anomaly-detection pipeline in
`hsi_workflow/`. Kept for provenance only — do not cite as results of the
revised objective (`Revised Research Objective.md`).

| Folder | Produced by | Notes |
|---|---|---|
| `composition/` | `legacy/composition_pipeline.py` | PCA+KMeans class/composition maps (the removed composition→FEA framing) |
| `unsupervised/` | `legacy/unsupervised_defect.py` | RX/Mahalanobis "defect" maps for LIG |
| `hsi_explore/` | `legacy/hsi_explore.py` | early quick-look QC figures (LIG-era) |
| `demo/`, `demo1/` | `legacy/hsi_explore.py --demo` | **synthetic** test cubes — not real data |

Current results live under `../workflow/`.
