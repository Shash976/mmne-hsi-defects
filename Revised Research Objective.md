# **Revised Research Objective**

Develop an unsupervised hyperspectral imaging framework for identifying and spatially localizing spectrally anomalous regions in SiO₂ thin films without requiring prior knowledge of defect type, location, or representative reference spectra..

We are **not detecting defects**.

We are detecting **spectral anomalies**.

# **Complete Pipeline**

# **Sample Inventory**

## **Input**

* Bare Si pieces  
* Processed SiO₂ pieces

---

### **Deliverables**

Sample database

| ID | Material | Dimensions | Notes |
| :---: | :---: | :---: | :---: |

---

### **Metrics**

* Number of samples  
* Sample size  
* Imaging area

Expected

10–20 independent specimens.

—----------------------------------------------------------------------------------------------------------------------------

## **Stage 1 — Data Acquisition**

SiO₂ Thin Film

↓

Hyperspectral Image Acquisition

↓

Raw Hyperspectral Cube  
(X,Y,λ)

Each pixel contains a reflectance spectrum.

Image every specimen.

Acquire

* Dark image  
* White reference  
* Sample

—----------------------------------------------------------------------------------------------------------------------------

# **The role of the pure silicon pieces**

The silicon pieces are **not** your reference spectra.

Instead, they become **control samples**.

Think of your experiment like this.

| Sample | Purpose |
| ----- | ----- |
| Bare Si | Control / baseline optical response |
| SiO₂-coated patterned pieces | Unknown experimental samples |

The silicon tells you:

"What does a relatively uniform substrate look like under HSI?"

The SiO₂ pieces tell you:

"How does the optical response change when films and processing are present?"

This is much stronger than comparing unknown samples to unknown samples.

# **What is your data?**

Let's assume your camera captures something like

512 × 512 pixels

×

300 wavelengths

Then one image is

512 × 512 \= 262,144 spectra

Each spectrum is

\[λ1, λ2, λ3, ... λ300\]

So

1 Image

↓

262,144 spectra

If you image

20 pieces

you already have

\~5 million spectra

That sounds huge—but there's a catch.

---

Neighboring pixels are almost identical.

□□□□□

□□□□□

□□□□□

These pixels all come from the same physical region.

Training ML on every pixel would massively overestimate performance because of **spatial autocorrelation**.

This is called **data leakage**.

---

# **So what is the real sample?**

For semiconductor HSI, the better approach is

Image

↓

Superpixel / ROI

↓

Average Spectrum

↓

One ML sample

Instead of

262,144 samples

you may only have

150 meaningful samples

from one image.

Those are much more statistically independent.

---

# **What should an ROI be?**

For your project

I'd avoid manually selecting defects because you don't know where they are.

Instead divide every image into fixed patches.

Example

1024 × 1024 image

↓

32 × 32 patches

Each patch becomes

ROI

For each ROI compute

* Mean spectrum  
* Standard deviation  
* PCA scores  
* Texture features (optional)

That ROI becomes **one sample**.

---

# **How many ROIs per image?**

A good target is

100–300 ROIs

per wafer piece.

Example

15 SiO₂ pieces

×

150 ROIs

\=

2250 samples

This is a good dataset for unsupervised learning.

---

# **How many images?**

From your photo I estimate

You have approximately

* 8–10 bare Si pieces  
* 20–30 processed SiO₂ pieces

(You'll know the exact count.)

I would image **every piece**.

One image per piece is the absolute minimum, but it's better to acquire multiple fields of view.

For example:

Each piece

↓

Center

↓

Top edge

↓

Bottom edge

↓

Left edge

↓

Right edge

So

5 images / piece

If you have

20 SiO₂ pieces

↓

100 images

plus

10 silicon pieces

↓

50 images

Total

150 images

That is already a solid research dataset.

---

# **Why use bare silicon?**

Because anomaly detection needs to know what "normal" looks like.

Think of bare silicon as

Baseline Population

Not

Reference Spectra

Those are different.

---

# **How much bare silicon?**

I would aim for roughly

25–35%

Control

65–75%

Experimental

Example

10 silicon pieces

20 SiO₂ pieces

or

50 silicon images

100 SiO₂ images

This ratio gives you enough baseline data without dominating the dataset.

---

# **Dataset size I would recommend**

Given your 20-day timeline and likely computational resources, here's a realistic target:

| Component | Target |
| ----- | ----- |
| Bare silicon pieces | **8–10** |
| Processed SiO₂ pieces | **20–30** |
| Images per piece | **3–5** |
| Total hyperspectral images | **90–200** |
| ROIs per image | **100–200** |
| Total ROI samples | **9,000–40,000** |
| Spectrum length | **\~200–300 wavelengths** (depending on your camera) |

This is more than enough for unsupervised anomaly detection.

---

## **One recommendation that will make your project much stronger**

Instead of treating every ROI independently, organize your dataset hierarchically:

Specimen

    │

    ├── Image 1

    │      ├── ROI 1

    │      ├── ROI 2

    │      └── ...

    │

    ├── Image 2

    │      ├── ROI 1

    │      └── ...

    │

    └── ...

Then, when you evaluate or compare methods, **split your data by specimen**, not by ROI. For example, if one SiO₂ fragment is held out for testing, none of its ROIs should appear in the training set. This prevents information leakage and gives you a much more realistic assessment of how well your anomaly detection pipeline generalizes to **new semiconductor samples**, which is ultimately the capability you'd want in a metrology workflow.

---

### **Deliverables**

Hyperspectral cube

(x,y,λ)

for every specimen.

---

### **Metrics**

Spatial resolution

Example

512×512 pixels

Spectral resolution

Example

300 bands

Signal-to-noise ratio (SNR)

Higher is better.

---

**Stage 2 — Radiometric Calibration**

Convert detector counts into calibrated reflectance.

Acquire

* Dark reference  
* White reference  
* Sample

Compute

R=S−DW−DR=\\frac{S-D}{W-D}R=W−DS−D​

Output

Reflectance Cube

Convert

Detector Counts

↓

Reflectance

---

### **Deliverables**

Reflectance cube

---

### **Metrics**

Reflectance range

Expected

0

↓

1

Histogram should not saturate.

---

## **Stage 3 — Preprocessing**

Purpose:  
 Remove measurement artifacts while preserving physical spectral information.

### **3.1 Background Removal**

Mask

* empty regions  
* sample holder  
* air

Keep only

SiO₂ film  
---

### **3.2 Spectral Smoothing**

Savitzky-Golay filter

Purpose

* remove sensor noise  
* preserve spectral peaks

---

### **3.3 Spectral Normalization**

Use

Standard Normal Variate (SNV)

or

Vector Normalization

Purpose

Remove illumination differences.

---

### **3.4 Optional Baseline Correction**

Remove scattering offsets if necessary.

Output

Preprocessed Spectral Cube  
---

# **Stage 4 — Exploratory Spectral Visualization**

This stage is now extremely important because it replaces the missing reference library.

Generate

### **Mean spectrum**

Entire film

↓

Average reflectance

---

### **Band images**

Visualize several wavelengths individually.

Question

Do different wavelengths reveal different structures?

---

### **RGB composite**

False-color visualization

Purpose

Human inspection

---

### **Spectral variance map**

Identify regions with unusually high spectral variability.

---

### **ROI inspection**

Manually inspect

* corners  
* center  
* edges

without assuming they are defective.

Perform

* Mask background  
* SG smoothing  
* SNV normalization

---

### **Deliverables**

Clean spectral dataset

---

### **Metrics**

Noise reduction

Compare

Before

↓

After

using

Root Mean Square Noise

or

Spectral SNR

Expected

Reduced high-frequency noise while retaining spectral shape.

---

# **Stage 5 — Dimensionality Reduction**

Now apply PCA.

Input

300 wavelengths

↓

Output

PC1

PC2

PC3

Deliverables

* PCA scatter plot  
* Explained variance  
* Loading plots  
* PC score images

This stage answers

Is there anything interesting before ML?

Generate

## **A**

RGB composite

---

## **B**

Band images

---

## **C**

Average spectrum

for

* Bare silicon  
* Every SiO₂ sample

---

## **D**

Variance map

---

## **E**

Spectral histogram

---

### **Deliverables**

5–10 figures.

---

### **Metrics**

Measure

Spectral variance

Expected

Silicon

↓

Low variance

Processed SiO₂

↓

Higher variance

---

---

# **Stage 6 — Unsupervised Anomaly Discovery**

Instead of

Reference Spectra

↓

Linear Spectral Unmixing

we now have

PCA Features

↓

K-means

or

DBSCAN

or

Gaussian Mixture Model

Purpose

Find naturally occurring spectral populations.

Output

Cluster Label

for

every pixel

Input

Every pixel spectrum.

Output

PC1

PC2

PC3

---

### **Deliverables**

* Explained variance plot  
* PCA scatter plot  
* Loading plot  
* PC score images

---

### **Metrics**

Expected

PC1

Usually

70–95%

variance

PC2

5–20%

PC3

1–10%

If

PC1

explains

only

20%

there may be excessive noise or high complexity. 

---

# **Stage 7 — Spatial Mapping**

Now project cluster labels back onto the image.

Instead of

Pixel

↓

Spectrum

you obtain

Film

↓

Cluster Map

Example

Blue

Green

Red

Yellow

Each color

\=

spectrally distinct region.

Notice

We are NOT calling them

vacancies

grain boundaries

cracks

because we don't know.

Perform

* K-Means  
* DBSCAN

Compare

cluster stability.

---

### **Deliverables**

Cluster map

---

### **Metrics**

Silhouette Score

Expected

0.4–0.8

Higher

↓

better separation.

Davies–Bouldin Index

Lower

↓

better.

Calinski–Harabasz

Higher

↓

better.

---

# **Stage 8 — Anomaly Scoring**

This is the stage I would add because your project is about anomaly detection.

Instead of simply clustering,

compute anomaly scores.

Possible methods

* Local Outlier Factor (LOF)  
* Isolation Forest  
* One-Class SVM  
* Mahalanobis Distance

Output

Anomaly Heatmap

High values

↓

Spectrally unusual regions.

This is much stronger than only using PCA.

Now compare every spectrum against

the "normal" population.

Methods

* Isolation Forest  
* Local Outlier Factor  
* Mahalanobis Distance

---

### **Deliverables**

Anomaly map

---

### **Metrics**

Average anomaly score

Percentage of anomalous pixels

Example

2%

5%

10%

Spatial distribution

Random?

Localized?

Edges?

---

# **Stage 9 — Spatial Postprocessing**

Clusters contain isolated noisy pixels.

Apply

* Median filter  
* Morphological opening  
* Connected component analysis

Purpose

Remove isolated artifacts.

Remove isolated pixels.

---

### **Deliverables**

Clean anomaly map.

---

### **Metrics**

Connected component size

Expected

Noise

↓

single pixels

Real regions

↓

large connected regions.

---

# **Stage 10 — Quantitative Maps**

Generate

### **PCA Score Maps**

PC1

PC2

PC3

---

### **Cluster Maps**

Different spectral populations

---

### **Spectral Distance Map**

Distance from

global mean spectrum

---

### **Anomaly Probability Map**

Every pixel receives

0–1

anomaly score.

Now characterize every anomaly.

For each region measure

Area

Perimeter

Compactness

Average spectrum

Spectral variance

Distance from silicon baseline

PCA coordinates

Anomaly score

---

### **Deliverables**

Region table

| Region | Area | Mean Reflectance | Variance | Anomaly |
| :---: | :---: | :---: | :---: | :---: |

---

### **Metrics**

Largest anomaly

Average anomaly

Region count

Spectral distance

---

# **Stage 11 — Region Characterization**

Instead of saying

"This is a vacancy"

we say

Region A

has

* lower reflectance  
* higher spectral variance  
* isolated anomaly score

Region B

has

* smooth spectrum  
* low anomaly score

This is scientifically correct.

Now compare

Silicon

↓

Baseline

Processed films

↓

Unknown

Questions

Are anomalies

Localized?

Repeated?

Near patterned regions?

Near edges?

Random?

---

### **Deliverables**

Final report

Including

* RGB images  
* PCA  
* Cluster map  
* Anomaly map  
* Statistics

---

# **Stage 12 — Future Validation**

This becomes future work.

Representative anomalous regions may later be examined using

* SEM  
* AFM  
* Raman  
* XPS  
* TEM

to determine the physical origin of the observed spectral anomalies.

Notice

Validation becomes future work.

Not part of the current pipeline.

---

# **Remove FEA**

This is the biggest thing I would remove.

Your abstract currently says

HSI

↓

Composition Map

↓

FEA

↓

Stress

↓

Device Performance

But you don't have

* stress measurements  
* composition measurements  
* strain measurements  
* boundary conditions  
* material constants

There is **no physical basis** for an FEA model yet.

I'd remove the entire FEA section from the current project and state it as future work.

# **Final Workflow**

SiO₂ Thin Films  
        │  
        ▼  
Hyperspectral Image Acquisition  
        │  
        ▼  
Radiometric Calibration  
(Dark / White Correction)  
        │  
        ▼  
Background Removal  
        │  
        ▼  
Spectral Smoothing  
        │  
        ▼  
Normalization  
        │  
        ▼  
Exploratory Visualization  
(Band Images, RGB Composite, Mean Spectra)  
        │  
        ▼  
Principal Component Analysis (PCA)  
        │  
        ▼  
Unsupervised Clustering  
(K-means / DBSCAN / GMM)  
        │  
        ▼  
Anomaly Detection  
(Isolation Forest / LOF / Mahalanobis)  
        │  
        ▼  
Spatial Filtering  
        │  
        ▼  
Cluster & Anomaly Maps  
        │  
        ▼  
Quantitative Region Characterization  
        │  
        ▼  
Candidate Defect Regions  
        │  
        ▼  
Future Physical Validation  
(SEM / Raman / AFM)

**an unsupervised hyperspectral anomaly detection framework** for SiO₂ thin films, producing **spectral anomaly maps** and **candidate defect regions** without prior labels or reference spectra. It can then state that these regions are intended to guide future high-resolution characterization (SEM, AFM, Raman) rather than replacing it. 

---

# **The revised hypothesis**

Instead of

Detect defects in SiO₂

your hypothesis becomes

**Hyperspectral imaging can distinguish spectrally anomalous regions in processed SiO₂ thin-film samples relative to a spectrally homogeneous silicon baseline without requiring prior defect labels.**

Notice the emphasis on **baseline**, not **ground truth**.

---

**What values should you expect?**

This is where I want to be careful. There are **no universal "correct" values** for semiconductor HSI because they depend on:

* the camera's wavelength range (VNIR? SWIR?)  
* illumination  
* optical setup  
* film thickness  
* surface finish  
* processing history

So instead of absolute values, you should aim for **quality metrics**.

| Stage | Metric | What indicates success? |
| ----- | ----- | ----- |
| Calibration | Reflectance | Values mostly between 0 and 1 without clipping |
| Preprocessing | Noise | Reduced high-frequency noise while preserving spectral shape |
| Silicon baseline | Spectral variance | Low variance across homogeneous regions |
| SiO₂ samples | Spectral variance | Higher than silicon if processing introduces heterogeneity |
| PCA | Explained variance | PC1 \+ PC2 explaining a substantial fraction of variance (often \>80%, but dataset dependent) |
| Clustering | Silhouette score | Positive separation; higher values indicate more distinct clusters |
| Anomaly detection | Anomaly fraction | Small, localized regions rather than the entire sample being anomalous |
| Spatial filtering | Connected components | Stable contiguous regions rather than isolated noisy pixels |

---

# **What actually goes into your ML table?**

After preprocessing, each ROI becomes one row.

| ROI ID | Sample | Material | Mean Spectrum | PCA1 | PCA2 | PCA3 | Spectral Variance | Anomaly Score |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| ROI001 | Si\_01 | Silicon | 300 values | ... | ... | ... | ... | ? |
| ROI002 | SiO₂\_04 | SiO₂ | 300 values | ... | ... | ... | ... | ? |

Notice that **material** is known (silicon vs processed sample), but **defect labels are not**.

---

# **How does the anomaly detector learn?**

This is another subtle point.

If you use an algorithm like **Isolation Forest**, it doesn't need labels.

It simply asks:

Which ROIs are statistically different from the majority?

Similarly, if you use **Local Outlier Factor (LOF)**, it compares each ROI to its local neighborhood in feature space.

You don't need to tell it where the defects are.

---

# **The most important scientific contribution**

I actually think you're asking the wrong question.

The contribution is **not**:

"Find defects."

The contribution is:

**Can hyperspectral imaging identify statistically significant spectral anomalies in semiconductor thin-film samples without prior labels?**

If the answer is **yes**, you've established a **non-destructive screening method**. That is valuable because it can direct expensive follow-up techniques (SEM, AFM, Raman, TEM) to only the most interesting regions.

In other words, your system becomes an **AI-assisted triage tool for semiconductor metrology**, rather than a replacement for established characterization techniques. That framing is realistic, scientifically defensible, and closely aligned with how such methods are adopted in semiconductor manufacturing

