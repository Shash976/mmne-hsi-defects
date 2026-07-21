# Overview — What this project is and why

## The goal in one sentence

> Build an **unsupervised** hyperspectral framework that identifies and spatially
> localizes **spectrally anomalous regions** in SiO₂ thin films, without prior
> knowledge of defect type, location, or reference spectra.

This is a **screening / triage** tool, not a defect classifier. It answers "where
is something unusual?" so that expensive follow-up techniques (SEM, AFM, Raman,
TEM) can be aimed only at the interesting spots.

## What changed from the old plan

The project was re-scoped (see [`Revised Research Objective.md`](../Revised%20Research%20Objective.md)).
The earlier framing — and the earlier code in [`legacy/`](../legacy/) — is dropped:

| Old framing (removed) | New framing (this repo) |
|---|---|
| Detect **defects** | Detect **spectral anomalies** |
| Reference spectra + linear unmixing | No references — unsupervised anomaly detection |
| Composition maps → FEA → stress | **No FEA**, no composition, no stress modeling |
| Optical density as the analysis signal | **SG-smoothed + SNV reflectance** as the signal |
| Per-pixel ML (leaky) | **ROI patches** + specimen-level splits (leakage-free) |

Optical density (`optical_density.py`) still exists in the codebase but is **off
the default path**. FEA is explicitly future work.

## Why bare silicon matters

Anomaly detection needs a notion of "normal." Bare silicon pieces provide a
**baseline population** — a spectrally homogeneous control that tells us what a
uniform substrate looks like under this exact camera/illumination setup. They are
**not** reference spectra to subtract; they are a statistical control.

| Sample | Purpose |
|---|---|
| Bare Si | Control / baseline optical response (spectrally uniform) |
| Processed SiO₂ | Unknown experimental samples (screened for anomalies) |

A key sanity check falls out of this: silicon should show **low** spectral
variance and processed SiO₂ **higher** variance (processing introduces
heterogeneity). `run_explore` persists this check as `material_variance.csv`
(pass both a silicon and a SiO₂ preset to get the comparison in one run).

## The revised hypothesis

> **Hyperspectral imaging can distinguish spectrally anomalous regions in
> processed SiO₂ thin-film samples relative to a spectrally homogeneous silicon
> baseline, without requiring prior defect labels.**

The emphasis is on **baseline**, not **ground truth**. If the answer is "yes,"
we've established a **non-destructive screening method** — an AI-assisted triage
tool for semiconductor metrology.

**How the pipeline operationalizes this** (important nuance): fitting detectors
directly on silicon and scoring SiO₂ flags ~100% of the film — silicon and SiO₂
are simply different materials, so the literal comparison degenerates into a
material classifier. The pipeline therefore produces **two products every run**:
the *silicon-baseline contrast map* (the hypothesis's literal comparison, kept as
its own deliverable) and the *within-film anomaly maps* (detectors fit on the
film's own majority) which drive the flagged regions — matching the document's
operational metrics ("small localized regions", 2–10% anomalous, "different from
the majority"). See [analysis.md](analysis.md#what-is-normal-both-answers-every-run-read-this).

## Why "no labels" is fine

Unsupervised outlier detectors (Isolation Forest, LOF, Mahalanobis) don't need to
be told where defects are. They ask a self-contained question:

> Which pixels/ROIs are statistically different from the **majority**?

The majority of a film is normal, so the oddballs surface on their own. See
[analysis.md](analysis.md) for exactly how "normal" is defined (the `fit_on`
setting) and why it matters.

## What "success" looks like

There are **no universal correct numbers** for semiconductor HSI (they depend on
camera range, illumination, film thickness, processing). So we track *quality
signals* rather than absolute values:

| Stage | Signal | Good outcome |
|---|---|---|
| Calibration | Reflectance | Mostly within 0–1, no clipping |
| Preprocessing | Noise | High-frequency noise reduced, spectral shape preserved |
| Silicon baseline | Spectral variance | Low, uniform |
| SiO₂ samples | Spectral variance | Higher than silicon |
| PCA | Explained variance | PC1 explains a large fraction (we saw ~84%) |
| Clustering | Silhouette | Positive, clusters distinct (we saw 0.33–0.53) |
| Anomaly | Anomalous fraction | Small and localized, not the whole sample |
| Spatial cleanup | Components | Contiguous regions, not isolated speckle |

## Where to go next

- The full stage-by-stage walkthrough: [pipeline.md](pipeline.md)
- How the code is organized and extended: [architecture.md](architecture.md)
- Running it: [usage.md](usage.md)
