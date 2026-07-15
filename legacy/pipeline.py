# hsi_pipeline.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report

def load_roi_spectra(roi_paths):
    """
    Each ROI -> (n_pixels, n_energy_channels) array + a label
    (pristine / grain_boundary / strained) + a film_id for grouping.
    Returns X (all pixels stacked), y (labels), groups (film_id per pixel).
    """
    X, y, groups = [], [], []
    for roi in roi_paths:
        spectra, label, film_id = read_roi(roi)   # your I/O logic
        X.append(spectra)
        y.append(np.full(len(spectra), label))
        groups.append(np.full(len(spectra), film_id))
    return np.vstack(X), np.concatenate(y), np.concatenate(groups)

def to_optical_density(raw_intensity, background_spectrum):
    return -np.log(raw_intensity / background_spectrum)

def make_synthetic_rois(pristine_spectrum, defect_spectrum, n=1000, seed=0):
    """
    Linear mixing + Gaussian blur + noise, for augmentation only.
    Keep these OUT of the validation set entirely.
    """
    rng = np.random.default_rng(seed)
    mixes = rng.uniform(0, 1, size=n)
    synth = np.outer(mixes, pristine_spectrum) + np.outer(1 - mixes, defect_spectrum)
    noise = rng.normal(0, 0.03, synth.shape)  # 2-5% noise
    return synth + noise, mixes

def build_pipeline(n_components=8):
    pca = PCA(n_components=n_components)
    models = {
        "plsda": PLSRegression(n_components=n_components),
        "rf": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=0),
        "svm": SVC(kernel="rbf", C=10, gamma="scale"),
    }
    return pca, models

def evaluate_leave_one_film_out(X, y, groups, pca, models):
    logo = LeaveOneGroupOut()
    X_reduced = pca.fit_transform(X)
    results = {name: [] for name in models}
    for train_idx, test_idx in logo.split(X_reduced, y, groups):
        for name, model in models.items():
            model.fit(X_reduced[train_idx], y[train_idx])
            preds = model.predict(X_reduced[test_idx])
            results[name].append(classification_report(y[test_idx], preds, output_dict=True))
    return results