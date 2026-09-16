# -*- coding: utf-8 -*-
"""Probabilistic failure-pressure modelling for corroded pipelines.

The pipeline compares native probabilistic TabPFN regression, a deep-kernel
Gaussian process (DKL-GP), conditional diffusion regression, and a deterministic
RSTRENG engineering benchmark on an applicable rectangular-defect subset.

Reproducibility choices preserved by this implementation include the documented
80/20 holdout split and seed, model-specific feature handling, and the reported
DKL-GP and diffusion architectures. Probabilistic outputs use model-native or
empirical quantiles where appropriate rather than imposing Gaussian intervals on
TabPFN or diffusion predictions.

Top-level switches:
    USE_IS_IRREGULAR
    USE_FEATURE_SELECTION

Outputs include tabular summaries, reproducibility records, and publication-
quality figures in TIF, PDF, SVG, and PNG formats.
"""

# Install project dependencies with:
#     pip install -r requirements.txt

import os
import sys
import math
import json
import random
import re
import hashlib
import platform
import shutil
import warnings
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import metadata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from scipy.stats import norm
import joblib

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression

import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.utils.data import TensorDataset, DataLoader

import gpytorch
from tabpfn import TabPFNRegressor

# Silence only known third-party deprecation noise. These are warnings, not errors.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*format of the columns of the 'remainder' transformer.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*Passing `palette` without assigning `hue`.*",
)
# Some TabPFN/scikit-learn calls emit the same warning from nested modules.
warnings.simplefilter("ignore", category=FutureWarning)

# =============================================================================
# USER SWITCHES
# =============================================================================
SEED = 123
USE_IS_IRREGULAR = True
USE_FEATURE_SELECTION = False

DATA_FILE = "Data.xlsx"
DATA_SOURCE_NAME = "Cai et al. public pipeline burst-pressure dataset"
DATA_SOURCE_CITATION = (
    "Cai, J., Jiang, X., Yang, Y., Lodewijks, G., and Wang, M. (2022). "
    "Data-driven Methods to Predict the Burst Strength of Corroded Line Pipelines "
    "Subjected to Internal Pressure. Journal of Marine Science and Application, "
    "21(2), 115-132. DOI: 10.1007/s11804-022-00263-0."
)
DATA_SOURCE_URL = "https://github.com/jiejie168/pipeBurstPressure_prediction_dataDriven"
SELECTED_COLUMN_POSITIONS = [2, 4, 5, 6, 9, 10, 13]

# if feature selection is ON, these settings are used for DKL / Diffusion
FS_MI_THRESHOLD = 0.01
FS_MAX_FEATURES = 10

# Probabilistic output settings
N_PREDICTIVE_SAMPLES = 1000       # final train/test distributions
N_CV_PREDICTIVE_SAMPLES = 300     # supplementary group-CV distributions
TABPFN_N_ESTIMATORS = 8
EXPECTED_TABPFN_VERSION = "8.1.0"  # version used for the reported reference results
DKL_SEED = 123
DIFFUSION_SEED = 123
NOMINAL_COVERAGES = (0.50, 0.80, 0.90, 0.95)
QUANTILES_TO_SAVE = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)


# RSTRENG engineering benchmark settings
# The benchmark is deliberately restricted to single, longitudinal, machined rectangular
# defects for which a uniform-depth axial profile can be reconstructed unambiguously
# from the reported corrosion length and depth.
RUN_RSTRENG_BENCHMARK = True
RSTRENG_PROFILE_CELLS = 100
RSTRENG_KSI_TO_MPA = 6.894757293168361
RSTRENG_FLOW_STRESS_INCREMENT_MPA = 10.0 * RSTRENG_KSI_TO_MPA

# Standard API 5L specified minimum yield strengths (SMYS), in ksi.
# These are used for the primary RSTRENG comparison so that the benchmark follows
# the standard flow-stress definition sigma_f = SMYS + 10 ksi.
RSTRENG_SMYS_KSI_BY_GRADE = {
    "A25": 25.0,
    "B": 35.0,
    "X42": 42.0,
    "X46": 46.0,
    "X52": 52.0,
    "X56": 56.0,
    "X60": 60.0,
    "X65": 65.0,
    "X80": 80.0,
}

RSTRENG_EXPECTED_HOLDOUT_INDICES = {2, 4, 23, 36, 43, 46, 60, 102}

PUBLICATION_SOURCE_NAMES = {
    1: "Benjamin et al. (2000)",
    2: "Cronin et al. (1996)",
    3: "Mok et al. (1991)",
    4: "Freire et al. (2006)",
    5: "Benjamin et al. (2005)",
    6: "Choi et al. (2003)",
    7: "Astanin et al. (2009)",
    8: "Cronin and Pick (2000)",
    9: "Chauhan et al. (2009)",
}

# =============================================================================
# Reproducibility + style
# =============================================================================
def set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


@contextmanager
def temporary_torch_seed(seed):
    """Use a local prediction seed without changing later model initialization."""
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


set_all_seeds(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    actual_tabpfn_version = metadata.version("tabpfn")
except metadata.PackageNotFoundError:
    actual_tabpfn_version = "not installed"
if actual_tabpfn_version != EXPECTED_TABPFN_VERSION:
    print(
        f"REPRODUCIBILITY NOTICE: TabPFN {actual_tabpfn_version} is installed; "
        f"the reported reference environment used {EXPECTED_TABPFN_VERSION}. "
        "Point predictions can change across TabPFN checkpoints/versions."
    )

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.18,
    "grid.linestyle": "--",
    "lines.linewidth": 1.8,
})
sns.set_theme(style="whitegrid", context="paper")

OUTPUT_DIR = f"model_results_minimal_probabilistic_seed_{SEED}_irr_{int(USE_IS_IRREGULAR)}_fs_{int(USE_FEATURE_SELECTION)}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 90)
print("RUN CONFIGURATION")
print("=" * 90)
print(f"SEED                 : {SEED}")
print(f"USE_IS_IRREGULAR     : {USE_IS_IRREGULAR}")
print(f"USE_FEATURE_SELECTION: {USE_FEATURE_SELECTION}")
print(f"DEVICE               : {DEVICE}")
print("=" * 90)

# =============================================================================
# Utilities
# =============================================================================
def save_figure(fig, filename_base):
    fig.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}.tif"), format="tiff",
                bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}.pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}.svg"), format="svg", bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}.png"), format="png", bbox_inches="tight")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def q_col(q):
    return f"Q{int(round(1000 * q)):03d}"


def crps_from_samples(y_true, samples):
    """Sample-based CRPS, one value per observation."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] != len(y_true):
        raise ValueError("samples must have shape (n_observations, n_draws)")
    term1 = np.mean(np.abs(samples - y_true[:, None]), axis=1)
    ordered = np.sort(samples, axis=1)
    m = ordered.shape[1]
    coeff = 2.0 * np.arange(1, m + 1) - m - 1.0
    half_pairwise = np.sum(ordered * coeff[None, :], axis=1) / (m**2)
    return term1 - half_pairwise


def interval_score(y_true, lower, upper, alpha):
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    score = upper - lower
    score += (2.0 / alpha) * (lower - y_true) * (y_true < lower)
    score += (2.0 / alpha) * (y_true - upper) * (y_true > upper)
    return score


def wis_from_bundle(y_true, bundle, coverages=NOMINAL_COVERAGES):
    y_true = np.asarray(y_true, dtype=float)
    total = 0.5 * np.abs(y_true - bundle.median)
    for coverage in coverages:
        alpha = 1.0 - coverage
        lower = bundle.quantile(alpha / 2.0)
        upper = bundle.quantile(1.0 - alpha / 2.0)
        total += (alpha / 2.0) * interval_score(y_true, lower, upper, alpha)
    return total / (len(coverages) + 0.5)


def gaussian_nll(y_true, mean_pred, std_pred, eps=1e-8):
    std_pred = np.maximum(np.asarray(std_pred, dtype=float), eps)
    y_true = np.asarray(y_true, dtype=float)
    mean_pred = np.asarray(mean_pred, dtype=float)
    return np.mean(
        0.5 * np.log(2 * np.pi * std_pred**2)
        + 0.5 * ((y_true - mean_pred) / std_pred) ** 2
    )


@dataclass
class PredictionBundle:
    """Common representation of each model's own predictive distribution."""
    mean: np.ndarray
    median: np.ndarray
    std: np.ndarray
    samples: np.ndarray
    quantiles: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)

    def quantile(self, q):
        q = float(q)
        if q in self.quantiles:
            return np.asarray(self.quantiles[q], dtype=float)
        return np.quantile(self.samples, q, axis=1)


def bundle_from_samples(samples, mean=None, median=None, std=None, quantiles=None, extras=None):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("Predictive samples must have shape (n_observations, n_draws)")
    return PredictionBundle(
        mean=np.asarray(samples.mean(axis=1) if mean is None else mean, dtype=float),
        median=np.asarray(np.median(samples, axis=1) if median is None else median, dtype=float),
        std=np.asarray(samples.std(axis=1, ddof=0) if std is None else std, dtype=float),
        samples=samples,
        quantiles={} if quantiles is None else {float(k): np.asarray(v, dtype=float) for k, v in quantiles.items()},
        extras={} if extras is None else dict(extras),
    )


def calibration_table(y_true, bundle, model_name, split_name):
    y_true = np.asarray(y_true, dtype=float)
    rows = []
    for coverage in NOMINAL_COVERAGES:
        alpha = 1.0 - coverage
        lower = bundle.quantile(alpha / 2.0)
        upper = bundle.quantile(1.0 - alpha / 2.0)
        covered = (y_true >= lower) & (y_true <= upper)
        rows.append({
            "Split": split_name,
            "Model": model_name,
            "Nominal_Coverage": coverage,
            "Empirical_Coverage": float(np.mean(covered)),
            "Coverage_Error": float(np.mean(covered) - coverage),
            "Mean_Interval_Width": float(np.mean(upper - lower)),
            "Median_Interval_Width": float(np.median(upper - lower)),
            "Mean_Interval_Score": float(np.mean(interval_score(y_true, lower, upper, alpha))),
        })
    return pd.DataFrame(rows)


def regression_metrics(y_true, bundle, gaussian_predictive=False):
    y_true = np.asarray(y_true, dtype=float)
    out = {
        "R2": r2_score(y_true, bundle.mean),
        "RMSE": rmse(y_true, bundle.mean),
        "MAE": mean_absolute_error(y_true, bundle.mean),
        "Median_RMSE": rmse(y_true, bundle.median),
        "Median_MAE": mean_absolute_error(y_true, bundle.median),
        "CRPS": float(np.mean(crps_from_samples(y_true, bundle.samples))),
        "WIS": float(np.mean(wis_from_bundle(y_true, bundle))),
        "Gaussian_NLL": float(gaussian_nll(y_true, bundle.mean, bundle.std)) if gaussian_predictive else np.nan,
    }
    for coverage in NOMINAL_COVERAGES:
        alpha = 1.0 - coverage
        lower = bundle.quantile(alpha / 2.0)
        upper = bundle.quantile(1.0 - alpha / 2.0)
        label = int(round(coverage * 100))
        out[f"PICP{label}"] = float(np.mean((y_true >= lower) & (y_true <= upper)))
        out[f"MPIW{label}"] = float(np.mean(upper - lower))
    return out


def select_features_train_only(
    X_train_df,
    y_train,
    X_test_df,
    mi_threshold=0.01,
    max_features=10,
    force_include=None,
    random_state=42,
):
    force_include = force_include or []

    mi = mutual_info_regression(X_train_df, y_train, random_state=random_state)
    mi_scores = pd.Series(mi, index=X_train_df.columns).sort_values(ascending=False)

    selected = mi_scores[mi_scores > mi_threshold].index.tolist()
    if len(selected) == 0:
        selected = mi_scores.head(max_features).index.tolist()
    if len(selected) > max_features:
        selected = selected[:max_features]

    for feat in force_include:
        if feat in X_train_df.columns and feat not in selected:
            if len(selected) < max_features:
                selected.append(feat)
            else:
                selected[-1] = feat

    selected = list(dict.fromkeys(selected))

    return (
        X_train_df[selected].copy(),
        X_test_df[selected].copy(),
        selected,
        mi_scores,
    )


def custom_permutation_importance_rmse(model_predict_fn, X, y, feature_names, n_repeats=25, random_state=42):
    rng = np.random.default_rng(random_state)
    baseline = rmse(y, model_predict_fn(X))
    importances = np.zeros((n_repeats, X.shape[1]))

    for r in range(n_repeats):
        for j in range(X.shape[1]):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            score = rmse(y, model_predict_fn(Xp))
            importances[r, j] = score - baseline

    return pd.DataFrame({
        "Feature": feature_names,
        "Importance_Mean": importances.mean(axis=0),
        "Importance_STD": importances.std(axis=0)
    }).sort_values("Importance_Mean", ascending=False)



# =============================================================================
# RSTRENG benchmark utilities
# =============================================================================
def _rstreng_text(value):
    """Normalize spreadsheet metadata for transparent eligibility screening."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _rstreng_parse_grade(specimen_id):
    """Extract API 5L grade from IDs such as 'X60,TS2.2' or 'B,kv43'."""
    if pd.isna(specimen_id):
        return ""
    return str(specimen_id).split(",")[0].strip().upper()


def rstreng_bulging_factor(length_mm, diameter_mm, thickness_mm):
    """
    RSTRENG/Effective-Area bulging factor used in the validated implementation.
    All geometric inputs are in mm.
    """
    length_mm = float(length_mm)
    diameter_mm = float(diameter_mm)
    thickness_mm = float(thickness_mm)

    if length_mm <= 0 or diameter_mm <= 0 or thickness_mm <= 0:
        raise ValueError("RSTRENG geometry must be positive.")

    l2dt = (length_mm ** 2) / (diameter_mm * thickness_mm)
    if l2dt <= 50.0:
        inside = 1.0 + 0.6275 * l2dt - 0.003375 * (l2dt ** 2)
        if inside <= 0:
            raise ValueError("Invalid RSTRENG bulging-factor radicand.")
        return math.sqrt(inside), 1, l2dt
    return 3.3 + 0.032 * l2dt, 2, l2dt


def rstreng_effective_area_search(depths_mm, lengths_mm, diameter_mm, thickness_mm):
    """
    Deterministic RSTRENG effective-area search.

    This reproduces the same RSF logic used in the validated RSTRENG implementation:
    every contiguous axial segment is examined and the minimum positive RSF governs.

    Parameters
    ----------
    depths_mm : array-like
        Corrosion depth assigned to each axial cell [mm].
    lengths_mm : array-like
        Axial length of each cell [mm].
    diameter_mm : float
        Pipe outside diameter [mm].
    thickness_mm : float
        Nominal wall thickness [mm].
    """
    depths = np.asarray(depths_mm, dtype=float).reshape(-1)
    lengths = np.asarray(lengths_mm, dtype=float).reshape(-1)

    if depths.size == 0 or depths.size != lengths.size:
        raise ValueError("depths_mm and lengths_mm must be nonempty arrays of equal length.")
    if np.any(~np.isfinite(depths)) or np.any(~np.isfinite(lengths)):
        raise ValueError("RSTRENG profile contains non-finite values.")
    if np.any(depths < 0) or np.any(lengths <= 0):
        raise ValueError("RSTRENG depths must be nonnegative and cell lengths positive.")
    if diameter_mm <= 0 or thickness_mm <= 0:
        raise ValueError("RSTRENG diameter and thickness must be positive.")

    # Prefix sums preserve the exact contiguous-segment logic while avoiding repeated slicing/summing.
    prefix_length = np.concatenate([[0.0], np.cumsum(lengths)])
    prefix_area = np.concatenate([[0.0], np.cumsum(depths * lengths)])

    min_rsf = float("inf")
    critical_depth = np.nan
    critical_length = np.nan
    critical_M = np.nan
    critical_bulging_type = np.nan
    critical_l2dt = np.nan
    critical_start = None
    critical_end = None

    n = len(depths)
    for p in range(n):
        for q in range(p, n):
            length_comb = prefix_length[q + 1] - prefix_length[p]
            area_comb = prefix_area[q + 1] - prefix_area[p]

            if length_comb <= 0:
                continue

            d_avg = area_comb / length_comb
            if d_avg >= thickness_mm:
                continue

            M, bulging_type, l2dt = rstreng_bulging_factor(
                length_comb, diameter_mm, thickness_mm
            )

            denominator = 1.0 - d_avg / (M * thickness_mm)
            if denominator <= 0:
                continue

            rsf = (1.0 - d_avg / thickness_mm) / denominator

            if 0.0 < rsf < min_rsf:
                min_rsf = rsf
                critical_depth = d_avg
                critical_length = length_comb
                critical_M = M
                critical_bulging_type = bulging_type
                critical_l2dt = l2dt
                critical_start = p
                critical_end = q

    if not np.isfinite(min_rsf):
        raise ValueError("No valid positive RSTRENG RSF was found for this profile.")

    return {
        "RSF": float(min_rsf),
        "Critical_Depth_mm": float(critical_depth),
        "Critical_Length_mm": float(critical_length),
        "Bulging_Factor_M": float(critical_M),
        "Bulging_Factor_Type": int(critical_bulging_type),
        "L2_over_Dt": float(critical_l2dt),
        "Critical_Start_Cell": int(critical_start),
        "Critical_End_Cell": int(critical_end),
    }


def rstreng_pressure_from_reference_yield(
    thickness_mm,
    diameter_mm,
    reference_yield_mpa,
    rsf,
):
    """
    Burst pressure in MPa using sigma_f = reference_yield + 10 ksi.

    For the primary engineering benchmark, reference_yield_mpa is SMYS.
    A measured-yield version is also saved as a diagnostic only.
    """
    flow_stress_mpa = float(reference_yield_mpa) + RSTRENG_FLOW_STRESS_INCREMENT_MPA
    pressure_mpa = (
        2.0 * float(thickness_mm) * flow_stress_mpa / float(diameter_mm)
    ) * float(rsf)
    return float(pressure_mpa), float(flow_stress_mpa)


def rstreng_rectangular_profile_assessment(
    corrosion_length_mm,
    corrosion_depth_mm,
    diameter_mm,
    thickness_mm,
    reference_yield_mpa,
    n_cells=RSTRENG_PROFILE_CELLS,
):
    """
    Reconstruct a reported rectangular metal-loss feature as a constant-depth
    axial profile and process it through the full RSTRENG contiguous-span search.

    The discretization is only a numerical representation of the stated
    rectangular geometry; it does not introduce an assumed irregular morphology.
    """
    corrosion_length_mm = float(corrosion_length_mm)
    corrosion_depth_mm = float(corrosion_depth_mm)
    diameter_mm = float(diameter_mm)
    thickness_mm = float(thickness_mm)

    if corrosion_length_mm <= 0:
        raise ValueError("Corrosion length must be positive.")
    if corrosion_depth_mm <= 0:
        raise ValueError("Corrosion depth must be positive.")
    if corrosion_depth_mm >= thickness_mm:
        raise ValueError("Corrosion depth must be smaller than wall thickness.")
    if int(n_cells) < 1:
        raise ValueError("n_cells must be >= 1.")

    n_cells = int(n_cells)
    depths = np.full(n_cells, corrosion_depth_mm, dtype=float)
    cell_lengths = np.full(n_cells, corrosion_length_mm / n_cells, dtype=float)

    result = rstreng_effective_area_search(
        depths_mm=depths,
        lengths_mm=cell_lengths,
        diameter_mm=diameter_mm,
        thickness_mm=thickness_mm,
    )
    pressure_mpa, flow_stress_mpa = rstreng_pressure_from_reference_yield(
        thickness_mm=thickness_mm,
        diameter_mm=diameter_mm,
        reference_yield_mpa=reference_yield_mpa,
        rsf=result["RSF"],
    )
    result["Predicted_Burst_Pressure_MPa"] = pressure_mpa
    result["Flow_Stress_MPa"] = flow_stress_mpa
    result["Profile_Cells"] = n_cells
    result["Reported_Rectangular_Length_mm"] = corrosion_length_mm
    result["Reported_Rectangular_Depth_mm"] = corrosion_depth_mm
    result["Full_Reported_Length_Governs"] = bool(
        np.isclose(
            result["Critical_Length_mm"],
            corrosion_length_mm,
            rtol=1e-9,
            atol=max(1e-9, 1e-9 * corrosion_length_mm),
        )
    )
    return result


def rstreng_eligibility_from_raw_row(raw_row):
    """
    Determine whether a database record can be used for the same-specimen
    RSTRENG benchmark without inventing an unknown corrosion morphology.

    Inclusion requires:
      - reported rectangular shape,
      - longitudinal orientation,
      - positive length/depth with depth < wall thickness,
      - machined or spark-eroded fabrication description,
      - no metadata indicating interacting/multiple defects,
      - a recognized material grade for SMYS.
    """
    shape = _rstreng_text(raw_row.get("shape", ""))
    orientation = _rstreng_text(raw_row.get("orientation", ""))
    notch_situation = _rstreng_text(raw_row.get("notch_situation", ""))
    remarks = _rstreng_text(raw_row.get("remarks", ""))
    grade = _rstreng_parse_grade(raw_row.get("ID", ""))

    length_mm = pd.to_numeric(pd.Series([raw_row.get("notch_length", np.nan)]), errors="coerce").iloc[0]
    depth_mm = pd.to_numeric(pd.Series([raw_row.get("notch_depth", np.nan)]), errors="coerce").iloc[0]
    thickness_mm = pd.to_numeric(pd.Series([raw_row.get("thickness", np.nan)]), errors="coerce").iloc[0]

    if shape != "rectangular":
        return False, "Excluded: shape is not reported as rectangular.", grade
    if orientation != "longitudinal":
        return False, "Excluded: defect orientation is not longitudinal.", grade
    if not np.isfinite(length_mm) or length_mm <= 0:
        return False, "Excluded: corrosion length is missing or nonpositive.", grade
    if not np.isfinite(depth_mm) or depth_mm <= 0:
        return False, "Excluded: corrosion depth is missing or nonpositive.", grade
    if not np.isfinite(thickness_mm) or thickness_mm <= 0:
        return False, "Excluded: wall thickness is missing or nonpositive.", grade
    if depth_mm >= thickness_mm:
        return False, "Excluded: reported corrosion depth is not smaller than wall thickness.", grade

    fabricated_rectangle = (
        "machined" in notch_situation
        or "spark erosion" in notch_situation
    )
    if not fabricated_rectangle:
        return False, "Excluded: rectangular geometry is not clearly identified as machined/spark-eroded.", grade

    # Known spreadsheet indicators of multi-defect/interacting configurations.
    interaction_text = f"{notch_situation} {remarks}"
    if any(token in interaction_text for token in ("mutl", "multiple", "multi defect", "interact")):
        return False, "Excluded: metadata indicate an interacting/multiple-defect configuration.", grade

    # Mok et al. interacting-groove notes such as '2-152' and '2||381'.
    if re.search(r"\b2\s*(?:\|\||-)\s*\d+", remarks):
        return False, "Excluded: remarks indicate two adjacent/interacting grooves.", grade

    if grade not in RSTRENG_SMYS_KSI_BY_GRADE:
        return False, f"Excluded: no SMYS mapping is defined for material grade {grade!r}.", grade

    return True, "Included: single longitudinal machined rectangular defect.", grade


def deterministic_point_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "Mean_Error": np.nan}
    return {
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "RMSE": float(rmse(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Mean_Error": float(np.mean(y_pred - y_true)),
    }


# =============================================================================
# Data loading + feature engineering
# =============================================================================
def load_and_preprocess_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Could not find {DATA_FILE!r}. Place it in the working directory "
            "or update DATA_FILE to point to the workbook."
        )

    raw_df = pd.read_excel(DATA_FILE, header=0)
    if raw_df.shape[1] <= max(SELECTED_COLUMN_POSITIONS):
        raise ValueError(
            f"{DATA_FILE} has only {raw_df.shape[1]} columns, but column position "
            f"{max(SELECTED_COLUMN_POSITIONS)} is required."
        )

    standardized_names = [
        "Burst_Pressure", "UTS", "Outer_Diameter", "Wall_Thickness",
        "Corrosion_Length", "Corrosion_Depth", "Shape_Of_Notch",
    ]
    role_by_position = {
        2: "Target: measured burst pressure",
        4: "Raw model input: ultimate tensile strength",
        5: "Raw model input: outside diameter",
        6: "Raw model input: wall thickness",
        9: "Raw model input: corrosion length",
        10: "Raw model input: corrosion depth",
        13: "Categorical input used to derive Is_Irregular",
    }
    standard_by_position = dict(zip(SELECTED_COLUMN_POSITIONS, standardized_names))
    audit_rows = []
    for position, original_name in enumerate(raw_df.columns):
        used = position in SELECTED_COLUMN_POSITIONS
        audit_rows.append({
            "Original_Position_Zero_Based": position,
            "Original_Column_Name": str(original_name),
            "Standardized_Name": standard_by_position.get(position, ""),
            "Status": "USED" if used else "REMOVED / NOT USED",
            "Role_or_Reason": role_by_position.get(
                position, "Not selected for this analysis."
            ),
        })
    column_audit_df = pd.DataFrame(audit_rows)

    df = raw_df.iloc[:, SELECTED_COLUMN_POSITIONS].copy()
    df.columns = standardized_names
    df.insert(0, "Original_Index", raw_df.index.to_numpy(dtype=int))
    df.insert(1, "Sample_ID", [f"ROW_{i:04d}" for i in raw_df.index])

    # The documented analysis retains all observations. Fail explicitly rather
    # than silently changing the research sample when required values are invalid.
    required_numeric = [
        "Burst_Pressure", "UTS", "Outer_Diameter", "Wall_Thickness",
        "Corrosion_Length", "Corrosion_Depth",
    ]
    for col in required_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    cleaning_rows = []
    for _, row in df.iterrows():
        reasons = []
        for col in required_numeric:
            if pd.isna(row[col]) or not np.isfinite(float(row[col])):
                reasons.append(f"invalid {col}")
        if pd.isna(row["Shape_Of_Notch"]):
            reasons.append("missing Shape_Of_Notch")
        cleaning_rows.append({
            "Original_Index": int(row["Original_Index"]),
            "Sample_ID": row["Sample_ID"],
            "Status": "RETAINED" if not reasons else "INVALID - RUN STOPPED",
            "Reason": "; ".join(reasons),
        })
    cleaning_log_df = pd.DataFrame(cleaning_rows)
    invalid = cleaning_log_df[cleaning_log_df["Status"] != "RETAINED"]
    if not invalid.empty:
        display(invalid)
        raise ValueError(
            "Invalid required data were found. The run was stopped so the "
            "documented 104-observation analysis is not silently changed."
        )

    # Correct cleaning of categorical label
    df["Shape_Of_Notch"] = (
        df["Shape_Of_Notch"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    df["Is_Irregular"] = df["Shape_Of_Notch"].eq("irregular").astype(int)

    # Retain the eight split strata so seed=123 reproduces the documented
    # 80/20 holdout indices exactly. Also record the corrected nine publication
    # sources separately for transparent reporting and supplementary group-CV.
    split_clusters = np.zeros(len(df), dtype=int)
    split_cluster_ranges = [
        (0, 8), (8, 28), (28, 43), (43, 51),
        (51, 56), (56, 64), (64, 99), (99, 104),
    ]
    for cluster_id, (start, end) in enumerate(split_cluster_ranges, 1):
        split_clusters[start:end] = cluster_id
    df["Split_Cluster"] = split_clusters
    df["Cluster"] = split_clusters  # retained for exact compatibility

    publication_sources = np.zeros(len(df), dtype=int)
    publication_ranges = [
    (0, 8),      # Benjamin et al. (2000): 8
    (8, 28),     # Cronin et al. (1996): 20
    (28, 43),    # Mok et al. (1991): 15
    (43, 51),    # Freire et al. (2006): 8
    (51, 56),    # Benjamin et al. (2005): 5
    (56, 62),    # Choi et al. (2003): 6
    (62, 63),    # Astanin et al. (2009): 1
    (63, 99),    # Cronin and Pick (2000): 36
    (99, 104),   # Chauhan and Crossley (2009): 5
]
    for source_id, (start, end) in enumerate(publication_ranges, 1):
        publication_sources[start:end] = source_id
    df["Publication_Source"] = publication_sources

    # Feature engineering
    df["Corrosion_Ratio"] = df["Corrosion_Depth"] / df["Wall_Thickness"]
    df["D_t_Ratio"] = df["Outer_Diameter"] / df["Wall_Thickness"]
    df["Metal_Area"] = (
        np.pi * (df["Outer_Diameter"] / 2) ** 2
        - np.pi * (df["Outer_Diameter"] / 2 - df["Wall_Thickness"]) ** 2
    )

    df["UTS_Wall_Interaction"] = df["UTS"] * df["Wall_Thickness"]
    df["Corrosion_Area"] = df["Corrosion_Length"] * df["Corrosion_Depth"]
    df["D_t_CorrosionRatio"] = df["D_t_Ratio"] * df["Corrosion_Ratio"]

    df["Pressure_Factor"] = df["UTS"] * df["Wall_Thickness"] / df["Outer_Diameter"]
    df["Corrosion_Severity"] = df["Corrosion_Area"] / df["Metal_Area"]
    df["Thinness_Factor"] = df["Wall_Thickness"] / df["Outer_Diameter"]
    df["Log_Corrosion_Length"] = np.log1p(df["Corrosion_Length"])
    df["Log_Corrosion_Area"] = np.log1p(df["Corrosion_Area"])

    df["Depth_sq"] = df["Corrosion_Depth"] ** 2
    df["Length_sq"] = df["Corrosion_Length"] ** 2
    df["Depth_Length"] = df["Corrosion_Depth"] * df["Corrosion_Length"]
    df["Pressure_Severity"] = df["Pressure_Factor"] * df["Corrosion_Severity"]

    return df, raw_df, column_audit_df, cleaning_log_df


df, raw_df, column_audit_df, cleaning_log_df = load_and_preprocess_data()

print("\nCounts of Shape_Of_Notch:")
print(df["Shape_Of_Notch"].value_counts(dropna=False))
print("\nCounts of Is_Irregular:")
print(df["Is_Irregular"].value_counts(dropna=False))

# Final raw input set
raw_features = [
    "UTS",
    "Outer_Diameter",
    "Wall_Thickness",
    "Corrosion_Length",
    "Corrosion_Depth",
]

if USE_IS_IRREGULAR:
    raw_features.append("Is_Irregular")

# All three models use the same raw input set
base_features = raw_features.copy()
master_features = raw_features.copy()

target_col = "Burst_Pressure"
split_group_col = "Cluster"
group_col = "Publication_Source"

X_all = df[master_features].copy()
y_all = df[target_col].values.astype(np.float32)
groups_all = df[group_col].values
split_groups_all = df[split_group_col].values
indices = np.arange(len(df))

# =============================================================================
# Hold-out split
# =============================================================================
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=SEED,
    stratify=df[split_group_col]
)

# Regression guard: these are the documented seed-123 test indices.
# If this assertion fails, the primary comparison no longer uses the same split.
EXPECTED_TEST_INDICES = np.array([2, 8, 4, 23, 90, 9, 28, 75, 102, 41, 80, 84, 78, 36, 60, 43, 53, 25, 70, 46, 71])
if SEED == 123 and len(df) == 104:
    if not np.array_equal(test_idx, EXPECTED_TEST_INDICES):
        raise RuntimeError(
            "The holdout split changed unexpectedly. Do not compare these results "
            "with the reported reference analysis until the split is restored."
        )

X_train_df = X_all.iloc[train_idx].reset_index(drop=True)
X_test_df = X_all.iloc[test_idx].reset_index(drop=True)

y_train = y_all[train_idx]
y_test = y_all[test_idx]

groups_train = groups_all[train_idx]
groups_test = groups_all[test_idx]
split_groups_train = split_groups_all[train_idx]
split_groups_test = split_groups_all[test_idx]

print("\nHold-out test shape counts:")
print(df.iloc[test_idx]["Shape_Of_Notch"].value_counts(dropna=False))
print("\nHold-out test irregular counts:")
print(df.iloc[test_idx]["Is_Irregular"].value_counts(dropna=False))

# =============================================================================
# Feature preparation
# =============================================================================
# TabPFN gets full active feature set
X_train_tabpfn_df = X_train_df.copy()
X_test_tabpfn_df = X_test_df.copy()

if USE_FEATURE_SELECTION:
    force_include = [
        "UTS", "Outer_Diameter", "Wall_Thickness",
        "Corrosion_Length", "Corrosion_Depth",
        "Corrosion_Ratio", "D_t_Ratio"
    ]
    force_include = [f for f in force_include if f in X_train_df.columns]

    X_train_sel_df, X_test_sel_df, selected_features, mi_scores = select_features_train_only(
        X_train_df=X_train_df,
        y_train=y_train,
        X_test_df=X_test_df,
        mi_threshold=FS_MI_THRESHOLD,
        max_features=FS_MAX_FEATURES,
        force_include=force_include,
        random_state=SEED,
    )
else:
    X_train_sel_df = X_train_df.copy()
    X_test_sel_df = X_test_df.copy()
    selected_features = X_train_df.columns.tolist()
    mi_scores = pd.Series(index=X_train_df.columns, data=np.nan)

print("\nSelected features for DKL-GP / Diffusion:")
print(selected_features)

mi_scores.to_csv(os.path.join(OUTPUT_DIR, "mutual_information_scores_holdout.csv"), header=["MI_Score"])

# scaling
x_scaler_tabpfn = StandardScaler()
X_train_tabpfn = x_scaler_tabpfn.fit_transform(X_train_tabpfn_df)
X_test_tabpfn = x_scaler_tabpfn.transform(X_test_tabpfn_df)

x_scaler_sel = StandardScaler()
X_train_sel = x_scaler_sel.fit_transform(X_train_sel_df)
X_test_sel = x_scaler_sel.transform(X_test_sel_df)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

# =============================================================================
# Model 1: TabPFN
# =============================================================================
def _tabpfn_full_output(model, X):
    """Call TabPFN while suppressing the known sklearn remainder-column warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r".*format of the columns of the 'remainder' transformer.*",
        )
        return model.predict(np.asarray(X, dtype=np.float32), output_type="full")


def predict_tabpfn_bundle(model, X, n_samples=N_PREDICTIVE_SAMPLES):
    output = _tabpfn_full_output(model, X)
    required = {"mean", "median", "criterion", "logits"}
    missing = required.difference(output.keys())
    if missing:
        raise RuntimeError(
            f"TabPFN output_type='full' is missing {sorted(missing)}. "
            "Install the pinned TabPFN version from the requirements file."
        )

    criterion = output["criterion"]
    logits = output["logits"]
    # Deterministic stratified quantile draws from TabPFN's own distribution.
    probabilities = (np.arange(n_samples, dtype=float) + 0.5) / n_samples
    draws = []
    with torch.no_grad():
        for probability in probabilities:
            draws.append(as_numpy(criterion.icdf(logits, float(probability))).reshape(-1))
    samples = np.stack(draws, axis=1)

    native_quantiles = {
        float(q): as_numpy(criterion.icdf(logits, float(q))).reshape(-1)
        for q in sorted(set(QUANTILES_TO_SAVE).union(
            {(1.0-c)/2.0 for c in NOMINAL_COVERAGES},
            {1.0-(1.0-c)/2.0 for c in NOMINAL_COVERAGES},
        ))
    }
    return bundle_from_samples(
        samples=samples,
        mean=as_numpy(output["mean"]).reshape(-1),
        median=as_numpy(output["median"]).reshape(-1),
        quantiles=native_quantiles,
        extras={"mode": as_numpy(output.get("mode", output["median"])).reshape(-1)},
    )


def fit_tabpfn(X_train, y_train, n_estimators=TABPFN_N_ESTIMATORS):
    model = TabPFNRegressor(
        random_state=SEED,
        n_estimators=n_estimators,
        device=DEVICE,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r".*format of the columns of the 'remainder' transformer.*",
        )
        model.fit(np.asarray(X_train, dtype=np.float32), np.asarray(y_train, dtype=float))
    return model


# =============================================================================
# Model 2: Deep Kernel Gaussian Process
# =============================================================================
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 8),
        )

    def forward(self, x):
        return self.net(x)


class DKLGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=8)
        )

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.base_kernel(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_dkl_gp(X_train, y_train_scaled, epochs=250, lr=0.01):
    train_x = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    train_y = torch.tensor(y_train_scaled, dtype=torch.float32).to(DEVICE)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
    feature_extractor = FeatureExtractor(X_train.shape[1]).to(DEVICE)
    model = DKLGPModel(train_x, train_y, likelihood, feature_extractor).to(DEVICE)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    return model, likelihood


def predict_dkl_gp(model, likelihood, X, y_scaler, n_samples=N_PREDICTIVE_SAMPLES, seed=SEED):
    model.eval()
    likelihood.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(X_t))
        mean_scaled = pred_dist.mean
        std_scaled = pred_dist.stddev.clamp_min(1e-8)

        # Draw from the model's own observed predictive distribution without
        # changing the RNG state used by any later model training.
        with temporary_torch_seed(seed):
            draws_scaled = pred_dist.rsample(torch.Size([int(n_samples)]))

    mean = y_scaler.inverse_transform(mean_scaled.cpu().numpy().reshape(-1, 1)).ravel()
    std = std_scaled.cpu().numpy() * y_scaler.scale_[0]
    samples = draws_scaled.cpu().numpy().T * y_scaler.scale_[0] + y_scaler.mean_[0]

    # Exact marginal Gaussian quantiles from the DKL-GP predictive distribution.
    needed_q = sorted(set(QUANTILES_TO_SAVE).union(
        {(1.0-c)/2.0 for c in NOMINAL_COVERAGES},
        {1.0-(1.0-c)/2.0 for c in NOMINAL_COVERAGES},
    ))
    quantiles = {float(q): mean + norm.ppf(q) * std for q in needed_q}
    return bundle_from_samples(
        samples=samples,
        mean=mean,
        median=mean.copy(),
        std=std,
        quantiles=quantiles,
        extras={"analytic_std": std.copy()},
    )


# =============================================================================
# Model 3: Conditional Diffusion Regressor
# =============================================================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb_scale = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class ConditionalDiffusionRegressor(nn.Module):
    def __init__(self, input_dim, hidden=64, time_dim=32):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        self.x_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Linear(hidden + time_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, y_t, t):
        xh = self.x_net(x)
        th = self.time_embed(t)
        inp = torch.cat([xh, y_t.unsqueeze(1), th], dim=1)
        eps_hat = self.net(inp).squeeze(1)
        return eps_hat


@dataclass
class DiffusionSchedule:
    T: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2


def make_schedule(T=100, beta_start=1e-4, beta_end=2e-2, device=DEVICE):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def train_conditional_diffusion(X_train, y_train_scaled, epochs=700, batch_size=16, lr=2e-3):
    schedule = DiffusionSchedule()
    betas, alphas, alpha_bars = make_schedule(schedule.T, schedule.beta_start, schedule.beta_end)

    model = ConditionalDiffusionRegressor(input_dim=X_train.shape[1], hidden=64, time_dim=32).to(DEVICE)

    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            t_idx = torch.randint(0, schedule.T, (xb.size(0),), device=DEVICE)
            alpha_bar_t = alpha_bars[t_idx]
            eps = torch.randn_like(yb)

            y_t = torch.sqrt(alpha_bar_t) * yb + torch.sqrt(1 - alpha_bar_t) * eps
            t_norm = t_idx.float() / (schedule.T - 1)

            eps_hat = model(xb, y_t, t_norm)
            loss = Fnn.mse_loss(eps_hat, eps)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model


def predict_conditional_diffusion(
    model,
    X,
    y_scaler,
    n_samples=N_PREDICTIVE_SAMPLES,
    seed=SEED,
    sample_chunk_size=100,
):
    """Generate and retain samples from the study conditional diffusion model.

    Training architecture and reverse-process equations match the reported study. Sampling is
    vectorized in chunks only to make 1,000 journal-quality draws practical.
    """
    schedule = DiffusionSchedule()
    betas, alphas, alpha_bars = make_schedule(
        schedule.T, schedule.beta_start, schedule.beta_end
    )

    model.eval()
    X_base = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    n_obs = X_base.size(0)
    all_draws = []

    with temporary_torch_seed(seed), torch.no_grad():
        generated = 0
        while generated < n_samples:
            chunk = min(sample_chunk_size, n_samples - generated)
            X_rep = X_base.unsqueeze(0).repeat(chunk, 1, 1).reshape(chunk * n_obs, -1)
            y_t = torch.randn(chunk * n_obs, device=DEVICE)

            for t in reversed(range(schedule.T)):
                t_batch = torch.full(
                    (chunk * n_obs,), t, device=DEVICE, dtype=torch.long
                )
                t_norm = t_batch.float() / (schedule.T - 1)
                beta_t = betas[t]
                alpha_t = alphas[t]
                alpha_bar_t = alpha_bars[t]
                eps_hat = model(X_rep, y_t, t_norm)
                reverse_mean = (1.0 / torch.sqrt(alpha_t)) * (
                    y_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_hat
                )
                if t > 0:
                    y_t = reverse_mean + torch.sqrt(beta_t) * torch.randn_like(y_t)
                else:
                    y_t = reverse_mean

            all_draws.append(y_t.reshape(chunk, n_obs).cpu().numpy())
            generated += chunk

    samples_scaled = np.concatenate(all_draws, axis=0)  # draws x observations
    samples = (samples_scaled * y_scaler.scale_[0] + y_scaler.mean_[0]).T
    quantiles = {
        float(q): np.quantile(samples, q, axis=1)
        for q in sorted(set(QUANTILES_TO_SAVE).union(
            {(1.0-c)/2.0 for c in NOMINAL_COVERAGES},
            {1.0-(1.0-c)/2.0 for c in NOMINAL_COVERAGES},
        ))
    }
    return bundle_from_samples(samples=samples, quantiles=quantiles)

# =============================================================================
# Train all models on the documented primary holdout split
# =============================================================================
print("\n" + "=" * 90)
print("TRAINING MODELS ON THE DOCUMENTED 80/20 HOLDOUT SPLIT")
print("=" * 90)

# Native probabilistic TabPFN using the study feature-scaling configuration.
# are deliberately retained to preserve point-prediction comparability.
tabpfn_model = fit_tabpfn(X_train_tabpfn, y_train)
y_bundle_test_tabpfn = predict_tabpfn_bundle(
    tabpfn_model, X_test_tabpfn, n_samples=N_PREDICTIVE_SAMPLES
)
y_bundle_train_tabpfn = predict_tabpfn_bundle(
    tabpfn_model, X_train_tabpfn, n_samples=N_PREDICTIVE_SAMPLES
)
print("TabPFN done.")

# Original DKL-GP architecture and training settings. A model-specific seed
# removes the accidental dependence on whether FNO happened to run beforehand.
set_all_seeds(DKL_SEED)
dkl_model, dkl_likelihood = train_dkl_gp(
    X_train_sel, y_train_scaled, epochs=250, lr=0.01
)
y_bundle_train_dkl = predict_dkl_gp(
    dkl_model, dkl_likelihood, X_train_sel, y_scaler,
    n_samples=N_PREDICTIVE_SAMPLES, seed=SEED + 101,
)
y_bundle_test_dkl = predict_dkl_gp(
    dkl_model, dkl_likelihood, X_test_sel, y_scaler,
    n_samples=N_PREDICTIVE_SAMPLES, seed=SEED + 102,
)
print("DKL-GP done.")

# Original direct conditional diffusion architecture and training settings.
set_all_seeds(DIFFUSION_SEED)
diff_model = train_conditional_diffusion(
    X_train_sel, y_train_scaled, epochs=700, batch_size=16, lr=2e-3
)
y_bundle_train_diff = predict_conditional_diffusion(
    diff_model, X_train_sel, y_scaler,
    n_samples=N_PREDICTIVE_SAMPLES, seed=SEED + 201,
)
y_bundle_test_diff = predict_conditional_diffusion(
    diff_model, X_test_sel, y_scaler,
    n_samples=N_PREDICTIVE_SAMPLES, seed=SEED + 202,
)
print("Conditional diffusion done.")

train_predictions = {
    "TabPFN": y_bundle_train_tabpfn,
    "DKL_GP": y_bundle_train_dkl,
    "Diffusion": y_bundle_train_diff,
}
test_predictions = {
    "TabPFN": y_bundle_test_tabpfn,
    "DKL_GP": y_bundle_test_dkl,
    "Diffusion": y_bundle_test_diff,
}
model_display = {
    "TabPFN": "TabPFN",
    "DKL_GP": "DKL-GP",
    "Diffusion": "Conditional diffusion",
}
model_colors = {
    "TabPFN": "#1f77b4",
    "DKL_GP": "#2ca02c",
    "Diffusion": "#d95f02",
}
model_markers = {"TabPFN": "o", "DKL_GP": "^", "Diffusion": "D"}
# =============================================================================
# FEATURE-SET ABLATION ANALYSIS
# =============================================================================
# Keep USE_FEATURE_SELECTION=True in the main configuration.
# This block separately evaluates all feature configurations using
# the same hold-out split, model settings, and random seeds.

RUN_FEATURE_ABLATION = False

if RUN_FEATURE_ABLATION:

    # ---------------------------------------------------------
    # 1. Define the four feature configurations
    # ---------------------------------------------------------
    raw_features = [
        "UTS",
        "Outer_Diameter",
        "Wall_Thickness",
        "Corrosion_Length",
        "Corrosion_Depth",
    ]

    if USE_IS_IRREGULAR:
        raw_features.append("Is_Irregular")

    # Raw variables plus the core mechanics-informed descriptors.
    mechanics_features = raw_features + [
        "Corrosion_Ratio",
        "D_t_Ratio",
        "Thinness_Factor",
        "Metal_Area",
        "UTS_Wall_Interaction",
        "Pressure_Factor",
        "Corrosion_Area",
        "Corrosion_Severity",
        "D_t_CorrosionRatio",
        "Pressure_Severity",
    ]

    # Remove Depth_Length because it duplicates Corrosion_Area = l*d.
    full_engineered_features = [
        feature for feature in master_features
        if feature != "Depth_Length"
    ]

    selected_ablation_features = [
        feature for feature in selected_features
        if feature != "Depth_Length"
    ]

    feature_sets = {
        "Raw measured variables": list(dict.fromkeys(raw_features)),
        "Raw + mechanics-informed": list(dict.fromkeys(mechanics_features)),
        "Full engineered set": list(dict.fromkeys(full_engineered_features)),
        "MI-selected set": list(dict.fromkeys(selected_ablation_features)),
    }

    # Save the exact definitions used in the ablation analysis.
    feature_set_rows = []
    for set_name, features in feature_sets.items():
        for feature in features:
            feature_set_rows.append({
                "Feature_Set": set_name,
                "Feature": feature,
            })

    feature_set_definitions_df = pd.DataFrame(feature_set_rows)
    feature_set_definitions_df.to_csv(
        os.path.join(OUTPUT_DIR, "feature_ablation_definitions.csv"),
        index=False,
    )

    # ---------------------------------------------------------
    # 2. Train and evaluate every model-feature combination
    # ---------------------------------------------------------
    ablation_rows = []

    def record_ablation_result(model_name, feature_set_name, features, bundle):
        metrics = regression_metrics(
            y_test,
            bundle,
            gaussian_predictive=(model_name == "DKL-GP"),
        )

        ablation_rows.append({
            "Model": model_name,
            "Feature_Set": feature_set_name,
            "N_Features": len(features),
            "Features": ", ".join(features),
            "R2": metrics["R2"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "CRPS": metrics["CRPS"],
            "WIS": metrics["WIS"],
            "PICP95": metrics["PICP95"],
            "MPIW95": metrics["MPIW95"],
        })

    for feature_set_name, features in feature_sets.items():

        print("\n" + "=" * 90)
        print(f"ABLATION FEATURE SET: {feature_set_name}")
        print(f"Number of features: {len(features)}")
        print(features)
        print("=" * 90)

        X_ablation_train_df = (
            df.iloc[train_idx][features]
            .reset_index(drop=True)
        )
        X_ablation_test_df = (
            df.iloc[test_idx][features]
            .reset_index(drop=True)
        )

        # Fit scaling using training observations only.
        ablation_x_scaler = StandardScaler()
        X_ablation_train = ablation_x_scaler.fit_transform(
            X_ablation_train_df
        )
        X_ablation_test = ablation_x_scaler.transform(
            X_ablation_test_df
        )

        ablation_y_scaler = StandardScaler()
        y_ablation_train_scaled = ablation_y_scaler.fit_transform(
            y_train.reshape(-1, 1)
        ).ravel()

        # -----------------------------------------------------
        # TabPFN
        # -----------------------------------------------------
        set_all_seeds(SEED)

        ablation_tabpfn = fit_tabpfn(
            X_ablation_train,
            y_train,
            n_estimators=TABPFN_N_ESTIMATORS,
        )

        ablation_tabpfn_bundle = predict_tabpfn_bundle(
            ablation_tabpfn,
            X_ablation_test,
            n_samples=N_PREDICTIVE_SAMPLES,
        )

        record_ablation_result(
            model_name="TabPFN",
            feature_set_name=feature_set_name,
            features=features,
            bundle=ablation_tabpfn_bundle,
        )

        del ablation_tabpfn, ablation_tabpfn_bundle

        # -----------------------------------------------------
        # DKL-GP
        # -----------------------------------------------------
        set_all_seeds(DKL_SEED)

        ablation_dkl, ablation_dkl_likelihood = train_dkl_gp(
            X_ablation_train,
            y_ablation_train_scaled,
            epochs=250,
            lr=0.01,
        )

        ablation_dkl_bundle = predict_dkl_gp(
            ablation_dkl,
            ablation_dkl_likelihood,
            X_ablation_test,
            ablation_y_scaler,
            n_samples=N_PREDICTIVE_SAMPLES,
            seed=SEED + 102,
        )

        record_ablation_result(
            model_name="DKL-GP",
            feature_set_name=feature_set_name,
            features=features,
            bundle=ablation_dkl_bundle,
        )

        del (
            ablation_dkl,
            ablation_dkl_likelihood,
            ablation_dkl_bundle,
        )

        # -----------------------------------------------------
        # Conditional diffusion
        # -----------------------------------------------------
        set_all_seeds(DIFFUSION_SEED)

        ablation_diffusion = train_conditional_diffusion(
            X_ablation_train,
            y_ablation_train_scaled,
            epochs=700,
            batch_size=16,
            lr=2e-3,
        )

        ablation_diffusion_bundle = predict_conditional_diffusion(
            ablation_diffusion,
            X_ablation_test,
            ablation_y_scaler,
            n_samples=N_PREDICTIVE_SAMPLES,
            seed=SEED + 202,
        )

        record_ablation_result(
            model_name="Conditional diffusion",
            feature_set_name=feature_set_name,
            features=features,
            bundle=ablation_diffusion_bundle,
        )

        del ablation_diffusion, ablation_diffusion_bundle

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 3. Save and display the comparison
    # ---------------------------------------------------------
    feature_ablation_df = pd.DataFrame(ablation_rows)

    feature_ablation_df = feature_ablation_df.sort_values(
        ["Model", "RMSE"],
        ascending=[True, True],
    ).reset_index(drop=True)

    feature_ablation_df.to_csv(
        os.path.join(OUTPUT_DIR, "feature_set_ablation_results.csv"),
        index=False,
    )

    print("\nFEATURE-SET ABLATION RESULTS")
    display(
        feature_ablation_df[
            [
                "Model",
                "Feature_Set",
                "N_Features",
                "R2",
                "RMSE",
                "MAE",
                "CRPS",
                "WIS",
                "PICP95",
                "MPIW95",
            ]
        ].round(4)
    )
# =============================================================================
# Hold-out summaries and predictions
# =============================================================================
summary_rows = []
calibration_frames = []
for model_name, test_bundle in test_predictions.items():
    train_metrics = regression_metrics(
        y_train, train_predictions[model_name], gaussian_predictive=(model_name == "DKL_GP")
    )
    test_metrics = regression_metrics(
        y_test, test_bundle, gaussian_predictive=(model_name == "DKL_GP")
    )
    row = {"Model": model_display[model_name]}
    row.update({f"Train_{k}": v for k, v in train_metrics.items()})
    row.update({f"Test_{k}": v for k, v in test_metrics.items()})
    summary_rows.append(row)
    calibration_frames.append(calibration_table(y_train, train_predictions[model_name], model_display[model_name], "Train"))
    calibration_frames.append(calibration_table(y_test, test_bundle, model_display[model_name], "Test"))

summary_df = pd.DataFrame(summary_rows).sort_values("Test_R2", ascending=False).reset_index(drop=True)
calibration_df = pd.concat(calibration_frames, ignore_index=True)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_holdout_metrics.csv"), index=False)
calibration_df.to_csv(os.path.join(OUTPUT_DIR, "calibration_all_levels.csv"), index=False)

print("\nHOLD-OUT MODEL COMPARISON")
display(summary_df.round(4))
print("\nCALIBRATION BY NOMINAL LEVEL")
display(calibration_df.round(4))

# Detailed test predictions with each model's own native/empirical quantiles.
pred_df = pd.DataFrame({
    "Original_Index": test_idx,
    "True_Burst_Pressure": y_test,
    "Shape_Of_Notch": df.iloc[test_idx]["Shape_Of_Notch"].values,
    "Is_Irregular": df.iloc[test_idx]["Is_Irregular"].values,
    "Split_Cluster": df.iloc[test_idx]["Split_Cluster"].values,
    "Publication_Source": df.iloc[test_idx]["Publication_Source"].values,
    "Corrosion_Ratio": df.iloc[test_idx]["Corrosion_Ratio"].values,
    "D_t_Ratio": df.iloc[test_idx]["D_t_Ratio"].values,
})
train_pred_df = pd.DataFrame({
    "Original_Index": train_idx,
    "True_Burst_Pressure": y_train,
    "Split_Cluster": df.iloc[train_idx]["Split_Cluster"].values,
    "Publication_Source": df.iloc[train_idx]["Publication_Source"].values,
})

for name, bundle in test_predictions.items():
    label = model_display[name].replace("-", "_").replace(" ", "_")
    pred_df[f"{label}_Mean"] = bundle.mean
    pred_df[f"{label}_Median"] = bundle.median
    pred_df[f"{label}_Std"] = bundle.std
    for q in QUANTILES_TO_SAVE:
        pred_df[f"{label}_{q_col(q)}"] = bundle.quantile(q)
    pred_df[f"{label}_Lower95"] = bundle.quantile(0.025)
    pred_df[f"{label}_Upper95"] = bundle.quantile(0.975)

for name, bundle in train_predictions.items():
    label = model_display[name].replace("-", "_").replace(" ", "_")
    train_pred_df[f"{label}_Mean"] = bundle.mean
    train_pred_df[f"{label}_Median"] = bundle.median
    train_pred_df[f"{label}_Std"] = bundle.std
    for q in QUANTILES_TO_SAVE:
        train_pred_df[f"{label}_{q_col(q)}"] = bundle.quantile(q)

pred_df.to_csv(os.path.join(OUTPUT_DIR, "final_predictions.csv"), index=False)
train_pred_df.to_csv(os.path.join(OUTPUT_DIR, "training_predictions.csv"), index=False)
print("\nTEST PREDICTIONS")
display(pred_df.round(4))

# Retain full predictive samples outside Excel.
# Physical-validity diagnostic. Predictions are not clipped because clipping would
# change the original model; any nonpositive samples are reported transparently.
physical_rows = []
for split_name, bundles in [("Train", train_predictions), ("Test", test_predictions)]:
    for model_name, bundle in bundles.items():
        physical_rows.append({
            "Split": split_name,
            "Model": model_display[model_name],
            "Negative_or_Zero_Mean_Count": int(np.sum(bundle.mean <= 0)),
            "Negative_or_Zero_Median_Count": int(np.sum(bundle.median <= 0)),
            "Nonpositive_Sample_Fraction": float(np.mean(bundle.samples <= 0)),
            "Minimum_Predicted_Mean": float(np.min(bundle.mean)),
            "Minimum_Predictive_Sample": float(np.min(bundle.samples)),
        })
physical_validity_df = pd.DataFrame(physical_rows)
physical_validity_df.to_csv(os.path.join(OUTPUT_DIR, "physical_validity_diagnostics.csv"), index=False)
print("\nPHYSICAL-VALIDITY DIAGNOSTICS (no clipping applied)")
display(physical_validity_df.round(6))

np.savez_compressed(
    os.path.join(OUTPUT_DIR, "predictive_samples_holdout.npz"),
    test_indices=test_idx,
    train_indices=train_idx,
    y_test=y_test,
    y_train=y_train,
    TabPFN_test=y_bundle_test_tabpfn.samples,
    TabPFN_train=y_bundle_train_tabpfn.samples,
    DKL_GP_test=y_bundle_test_dkl.samples,
    DKL_GP_train=y_bundle_train_dkl.samples,
    Diffusion_test=y_bundle_test_diff.samples,
    Diffusion_train=y_bundle_train_diff.samples,
)


# =============================================================================
# RSTRENG engineering benchmark on the applicable rectangular-defect subset
# =============================================================================
# This benchmark is intentionally separate from the 21-specimen primary hold-out.
# Only cases whose reported metadata support an unambiguous uniform-depth
# rectangular axial profile are included. All four models are then compared on
# exactly the same eligible hold-out specimens.

rstreng_eligibility_rows = []
rstreng_applicable_rows = []
rstreng_holdout_comparison_rows = []
rstreng_same_subset_metrics_df = pd.DataFrame()
rstreng_benchmark_config_df = pd.DataFrame()

if RUN_RSTRENG_BENCHMARK:
    required_raw_columns = {
        "ID", "burst pressure", "yield stress", "outer diameter", "thickness",
        "notch_length", "notch_depth", "notch_situation", "orientation",
        "shape", "remarks",
    }
    missing_raw_columns = required_raw_columns.difference(raw_df.columns)
    if missing_raw_columns:
        raise KeyError(
            "RSTRENG benchmark requires the following original spreadsheet columns: "
            f"{sorted(missing_raw_columns)}"
        )

    test_index_set = {int(i) for i in test_idx}

    for original_index, raw_row in raw_df.iterrows():
        eligible, reason, grade = rstreng_eligibility_from_raw_row(raw_row)
        source_series = df.loc[
            df["Original_Index"].eq(int(original_index)), "Publication_Source"
        ]
        source_id = int(source_series.iloc[0]) if len(source_series) else np.nan

        audit_row = {
            "Original_Index": int(original_index),
            "Source_Row_Number": raw_row.get("num ", np.nan),
            "ID": raw_row.get("ID", ""),
            "Publication_Source": source_id,
            "Publication_Source_Name": PUBLICATION_SOURCE_NAMES.get(source_id, ""),
            "Material_Grade": grade,
            "Shape": raw_row.get("shape", ""),
            "Orientation": raw_row.get("orientation", ""),
            "Notch_Situation": raw_row.get("notch_situation", ""),
            "Remarks": raw_row.get("remarks", ""),
            "Corrosion_Length_mm": pd.to_numeric(
                pd.Series([raw_row.get("notch_length", np.nan)]), errors="coerce"
            ).iloc[0],
            "Corrosion_Depth_mm": pd.to_numeric(
                pd.Series([raw_row.get("notch_depth", np.nan)]), errors="coerce"
            ).iloc[0],
            "Wall_Thickness_mm": pd.to_numeric(
                pd.Series([raw_row.get("thickness", np.nan)]), errors="coerce"
            ).iloc[0],
            "Eligible_for_RSTRENG_Benchmark": bool(eligible),
            "Eligibility_Reason": reason,
            "In_Primary_Holdout": int(original_index) in test_index_set,
        }
        rstreng_eligibility_rows.append(audit_row)

        if not eligible:
            continue

        smys_ksi = RSTRENG_SMYS_KSI_BY_GRADE[grade]
        smys_mpa = smys_ksi * RSTRENG_KSI_TO_MPA
        measured_yield_mpa = float(raw_row["yield stress"])

        primary = rstreng_rectangular_profile_assessment(
            corrosion_length_mm=float(raw_row["notch_length"]),
            corrosion_depth_mm=float(raw_row["notch_depth"]),
            diameter_mm=float(raw_row["outer diameter"]),
            thickness_mm=float(raw_row["thickness"]),
            reference_yield_mpa=smys_mpa,
            n_cells=RSTRENG_PROFILE_CELLS,
        )

        # Diagnostic only: same geometry and RSF, but using measured yield stress + 10 ksi.
        # The standard-SMYS prediction above remains the primary reported RSTRENG benchmark.
        measured_yield_prediction, measured_yield_flow_stress = (
            rstreng_pressure_from_reference_yield(
                thickness_mm=float(raw_row["thickness"]),
                diameter_mm=float(raw_row["outer diameter"]),
                reference_yield_mpa=measured_yield_mpa,
                rsf=primary["RSF"],
            )
        )

        applicable_row = {
            "Original_Index": int(original_index),
            "Source_Row_Number": raw_row.get("num ", np.nan),
            "ID": raw_row["ID"],
            "Publication_Source": source_id,
            "Publication_Source_Name": PUBLICATION_SOURCE_NAMES.get(source_id, ""),
            "Material_Grade": grade,
            "Measured_Burst_Pressure_MPa": float(raw_row["burst pressure"]),
            "Measured_Yield_Stress_MPa": measured_yield_mpa,
            "SMYS_ksi": smys_ksi,
            "SMYS_MPa": smys_mpa,
            "Outer_Diameter_mm": float(raw_row["outer diameter"]),
            "Wall_Thickness_mm": float(raw_row["thickness"]),
            "Corrosion_Length_mm": float(raw_row["notch_length"]),
            "Corrosion_Depth_mm": float(raw_row["notch_depth"]),
            "RSTRENG_RSF": primary["RSF"],
            "RSTRENG_Critical_Depth_mm": primary["Critical_Depth_mm"],
            "RSTRENG_Critical_Length_mm": primary["Critical_Length_mm"],
            "RSTRENG_Bulging_Factor_M": primary["Bulging_Factor_M"],
            "RSTRENG_Bulging_Factor_Type": primary["Bulging_Factor_Type"],
            "RSTRENG_L2_over_Dt": primary["L2_over_Dt"],
            "RSTRENG_Flow_Stress_SMYSplus10ksi_MPa": primary["Flow_Stress_MPa"],
            "RSTRENG_Prediction_SMYSplus10ksi_MPa": primary["Predicted_Burst_Pressure_MPa"],
            "RSTRENG_Prediction_MeasuredYSplus10ksi_Diagnostic_MPa": measured_yield_prediction,
            "RSTRENG_Flow_Stress_MeasuredYSplus10ksi_Diagnostic_MPa": measured_yield_flow_stress,
            "RSTRENG_Profile_Cells": primary["Profile_Cells"],
            "RSTRENG_Full_Reported_Length_Governs": primary["Full_Reported_Length_Governs"],
            "In_Primary_Holdout": int(original_index) in test_index_set,
        }
        rstreng_applicable_rows.append(applicable_row)

    rstreng_eligibility_df = pd.DataFrame(rstreng_eligibility_rows)
    rstreng_all_applicable_df = pd.DataFrame(rstreng_applicable_rows)

    # Eligible indices in the exact primary hold-out order.
    eligible_index_set = set(
        rstreng_all_applicable_df.loc[
            rstreng_all_applicable_df["In_Primary_Holdout"], "Original_Index"
        ].astype(int)
    )
    rstreng_holdout_indices = [
        int(idx) for idx in test_idx if int(idx) in eligible_index_set
    ]

    if set(rstreng_holdout_indices) != RSTRENG_EXPECTED_HOLDOUT_INDICES:
        warnings.warn(
            "RSTRENG applicable hold-out subset differs from the expected audited set. "
            f"Expected {sorted(RSTRENG_EXPECTED_HOLDOUT_INDICES)}, "
            f"found {sorted(rstreng_holdout_indices)}. Review eligibility_audit.csv."
        )

    # Position lookup is required because model bundles are stored in test_idx order.
    test_position_by_original_index = {
        int(original_idx): position
        for position, original_idx in enumerate(test_idx)
    }

    applicable_by_index = (
        rstreng_all_applicable_df
        .set_index("Original_Index", drop=False)
    )

    for original_index in rstreng_holdout_indices:
        test_position = test_position_by_original_index[original_index]
        base = applicable_by_index.loc[original_index].to_dict()

        base.update({
            "TabPFN_Mean_MPa": float(y_bundle_test_tabpfn.mean[test_position]),
            "DKL_GP_Mean_MPa": float(y_bundle_test_dkl.mean[test_position]),
            "Conditional_Diffusion_Mean_MPa": float(y_bundle_test_diff.mean[test_position]),
            "TabPFN_Lower95_MPa": float(y_bundle_test_tabpfn.quantile(0.025)[test_position]),
            "TabPFN_Upper95_MPa": float(y_bundle_test_tabpfn.quantile(0.975)[test_position]),
            "DKL_GP_Lower95_MPa": float(y_bundle_test_dkl.quantile(0.025)[test_position]),
            "DKL_GP_Upper95_MPa": float(y_bundle_test_dkl.quantile(0.975)[test_position]),
            "Conditional_Diffusion_Lower95_MPa": float(y_bundle_test_diff.quantile(0.025)[test_position]),
            "Conditional_Diffusion_Upper95_MPa": float(y_bundle_test_diff.quantile(0.975)[test_position]),
        })
        rstreng_holdout_comparison_rows.append(base)

    rstreng_holdout_comparison_df = pd.DataFrame(rstreng_holdout_comparison_rows)

    if len(rstreng_holdout_comparison_df) == 0:
        raise RuntimeError(
            "No RSTRENG-applicable specimens were found in the primary hold-out."
        )

    # Preserve primary hold-out order in the exported comparison.
    order_lookup = {
        int(original_index): position
        for position, original_index in enumerate(rstreng_holdout_indices)
    }
    rstreng_holdout_comparison_df["_Holdout_Order"] = (
        rstreng_holdout_comparison_df["Original_Index"].map(order_lookup)
    )
    rstreng_holdout_comparison_df = (
        rstreng_holdout_comparison_df
        .sort_values("_Holdout_Order")
        .drop(columns="_Holdout_Order")
        .reset_index(drop=True)
    )

    y_rstreng_subset = rstreng_holdout_comparison_df["Measured_Burst_Pressure_MPa"].to_numpy(dtype=float)
    comparison_prediction_columns = [
        ("RSTRENG", "RSTRENG_Prediction_SMYSplus10ksi_MPa"),
        ("TabPFN", "TabPFN_Mean_MPa"),
        ("DKL-GP", "DKL_GP_Mean_MPa"),
        ("Conditional diffusion", "Conditional_Diffusion_Mean_MPa"),
    ]

    metric_rows = []
    for model_label, prediction_column in comparison_prediction_columns:
        y_pred_subset = rstreng_holdout_comparison_df[prediction_column].to_numpy(dtype=float)
        metrics = deterministic_point_metrics(y_rstreng_subset, y_pred_subset)
        metric_rows.append({
            "Model": model_label,
            "N": len(y_rstreng_subset),
            **metrics,
            "Mean_Predicted_to_Measured_Ratio": float(
                np.mean(y_pred_subset / y_rstreng_subset)
            ),
        })
    rstreng_same_subset_metrics_df = pd.DataFrame(metric_rows)

    # Add specimen-level errors/ratios for transparent plotting and audit.
    for model_label, prediction_column in comparison_prediction_columns:
        clean_label = (
            model_label.replace("-", "_")
            .replace(" ", "_")
            .replace("Conditional_diffusion", "Diffusion")
        )
        prediction = rstreng_holdout_comparison_df[prediction_column].to_numpy(dtype=float)
        truth = rstreng_holdout_comparison_df["Measured_Burst_Pressure_MPa"].to_numpy(dtype=float)
        rstreng_holdout_comparison_df[f"{clean_label}_Error_MPa"] = prediction - truth
        rstreng_holdout_comparison_df[f"{clean_label}_Absolute_Error_MPa"] = np.abs(prediction - truth)
        rstreng_holdout_comparison_df[f"{clean_label}_Predicted_to_Measured_Ratio"] = prediction / truth

    rstreng_benchmark_config_df = pd.DataFrame([
        ["Benchmark enabled", RUN_RSTRENG_BENCHMARK],
        ["Profile reconstruction", "Uniform-depth rectangular axial profile from reported length and depth"],
        ["Profile cells", RSTRENG_PROFILE_CELLS],
        ["Primary flow stress", "SMYS + 10 ksi"],
        ["Measured-yield version", "Saved as diagnostic only; not used for headline comparison"],
        ["Eligible shape", "Rectangular"],
        ["Eligible orientation", "Longitudinal"],
        ["Eligible fabrication", "Machined or spark-eroded"],
        ["Interacting/multiple defects", "Excluded"],
        ["Irregular/natural defects", "Excluded"],
        ["Intact specimens", "Excluded"],
        ["Expected primary-holdout eligible N", len(RSTRENG_EXPECTED_HOLDOUT_INDICES)],
        ["Observed primary-holdout eligible N", len(rstreng_holdout_comparison_df)],
    ], columns=["Setting", "Value"])

    rstreng_eligibility_df.to_csv(
        os.path.join(OUTPUT_DIR, "rstreng_eligibility_audit.csv"), index=False
    )
    rstreng_all_applicable_df.to_csv(
        os.path.join(OUTPUT_DIR, "rstreng_all_applicable_predictions.csv"), index=False
    )
    rstreng_holdout_comparison_df.to_csv(
        os.path.join(OUTPUT_DIR, "rstreng_same_holdout_subset_predictions.csv"), index=False
    )
    rstreng_same_subset_metrics_df.to_csv(
        os.path.join(OUTPUT_DIR, "rstreng_same_holdout_subset_metrics.csv"), index=False
    )
    rstreng_benchmark_config_df.to_csv(
        os.path.join(OUTPUT_DIR, "rstreng_benchmark_config.csv"), index=False
    )

    print("\n" + "=" * 90)
    print("RSTRENG ENGINEERING BENCHMARK")
    print("=" * 90)
    print(
        f"Applicable records in full 104-record database: "
        f"{int(rstreng_eligibility_df['Eligible_for_RSTRENG_Benchmark'].sum())}"
    )
    print(
        f"Applicable records in the fixed 21-record hold-out: "
        f"{len(rstreng_holdout_comparison_df)}"
    )
    print("\nSame-specimen hold-out subset:")
    display(
        rstreng_holdout_comparison_df[
            [
                "Original_Index",
                "ID",
                "Publication_Source_Name",
                "Measured_Burst_Pressure_MPa",
                "RSTRENG_Prediction_SMYSplus10ksi_MPa",
                "TabPFN_Mean_MPa",
                "DKL_GP_Mean_MPa",
                "Conditional_Diffusion_Mean_MPa",
            ]
        ].round(4)
    )
    print("\nSame-specimen point-prediction metrics:")
    display(rstreng_same_subset_metrics_df.round(4))

else:
    rstreng_eligibility_df = pd.DataFrame()
    rstreng_all_applicable_df = pd.DataFrame()
    rstreng_holdout_comparison_df = pd.DataFrame()
    rstreng_same_subset_metrics_df = pd.DataFrame()
    rstreng_benchmark_config_df = pd.DataFrame()


# =============================================================================
# Supplementary group-aware CV using corrected nine publication sources
# =============================================================================
print("\n" + "=" * 90)
print("SUPPLEMENTARY GROUP-AWARE CROSS-VALIDATION")
print("=" * 90)

gkf = GroupKFold(n_splits=min(5, len(np.unique(groups_all))))
cv_records = []
cv_prediction_rows = []
selected_features_by_fold = []

for fold, (tr, va) in enumerate(gkf.split(X_all, y_all, groups=groups_all), start=1):
    print(f"Fold {fold}/5")
    X_tr_df = X_all.iloc[tr].reset_index(drop=True)
    X_va_df = X_all.iloc[va].reset_index(drop=True)
    y_tr = y_all[tr]
    y_va = y_all[va]

    X_tr_tab_df = X_tr_df.copy()
    X_va_tab_df = X_va_df.copy()

    if USE_FEATURE_SELECTION:
        force_include = [
            "UTS", "Outer_Diameter", "Wall_Thickness",
            "Corrosion_Length", "Corrosion_Depth",
            "Corrosion_Ratio", "D_t_Ratio",
        ]
        force_include = [f for f in force_include if f in X_tr_df.columns]
        X_tr_sel_df_fold, X_va_sel_df_fold, selected_features_fold, _ = select_features_train_only(
            X_train_df=X_tr_df,
            y_train=y_tr,
            X_test_df=X_va_df,
            mi_threshold=FS_MI_THRESHOLD,
            max_features=FS_MAX_FEATURES,
            force_include=force_include,
            random_state=SEED + fold,
        )
    else:
        X_tr_sel_df_fold = X_tr_df.copy()
        X_va_sel_df_fold = X_va_df.copy()
        selected_features_fold = X_tr_df.columns.tolist()

    selected_features_by_fold.append({
        "Fold": fold,
        "Validation_Sources": ", ".join(map(str, sorted(np.unique(groups_all[va])))),
        "Selected_Features": ", ".join(selected_features_fold),
    })

    xsc_tab = StandardScaler()
    X_tr_tab = xsc_tab.fit_transform(X_tr_tab_df)
    X_va_tab = xsc_tab.transform(X_va_tab_df)
    xsc_sel = StandardScaler()
    X_tr_sel_fold = xsc_sel.fit_transform(X_tr_sel_df_fold)
    X_va_sel_fold = xsc_sel.transform(X_va_sel_df_fold)
    ysc = StandardScaler()
    y_tr_sc = ysc.fit_transform(y_tr.reshape(-1, 1)).ravel()

    # TabPFN
    model_tab = fit_tabpfn(X_tr_tab, y_tr)
    bundle_tab = predict_tabpfn_bundle(model_tab, X_va_tab, n_samples=N_CV_PREDICTIVE_SAMPLES)
    # DKL-GP
    set_all_seeds(DKL_SEED + fold)
    model_dkl_cv, lik_dkl_cv = train_dkl_gp(X_tr_sel_fold, y_tr_sc, epochs=180, lr=0.01)
    bundle_dkl = predict_dkl_gp(
        model_dkl_cv, lik_dkl_cv, X_va_sel_fold, ysc,
        n_samples=N_CV_PREDICTIVE_SAMPLES, seed=SEED + 1000 + fold,
    )
    # Diffusion
    set_all_seeds(DIFFUSION_SEED + fold)
    model_diff_cv = train_conditional_diffusion(X_tr_sel_fold, y_tr_sc, epochs=450)
    bundle_diff = predict_conditional_diffusion(
        model_diff_cv, X_va_sel_fold, ysc,
        n_samples=N_CV_PREDICTIVE_SAMPLES, seed=SEED + 2000 + fold,
    )

    fold_bundles = {"TabPFN": bundle_tab, "DKL_GP": bundle_dkl, "Diffusion": bundle_diff}
    for model_name, bundle in fold_bundles.items():
        rec = regression_metrics(y_va, bundle, gaussian_predictive=(model_name == "DKL_GP"))
        rec.update({
            "Model": model_display[model_name],
            "Fold": fold,
            "N_Train": len(tr),
            "N_Validation": len(va),
            "Validation_Sources": ", ".join(map(str, sorted(np.unique(groups_all[va])))),
        })
        cv_records.append(rec)
        for local_i, global_i in enumerate(va):
            cv_prediction_rows.append({
                "Fold": fold,
                "Model": model_display[model_name],
                "Original_Index": int(global_i),
                "Publication_Source": int(groups_all[global_i]),
                "True_Burst_Pressure": float(y_all[global_i]),
                "Predicted_Mean": float(bundle.mean[local_i]),
                "Predicted_Median": float(bundle.median[local_i]),
                "Predictive_STD": float(bundle.std[local_i]),
                "Lower95": float(bundle.quantile(0.025)[local_i]),
                "Upper95": float(bundle.quantile(0.975)[local_i]),
            })

cv_df = pd.DataFrame(cv_records)
cv_predictions_df = pd.DataFrame(cv_prediction_rows)
selected_features_by_fold_df = pd.DataFrame(selected_features_by_fold)
cv_summary = cv_df.groupby("Model", as_index=False).agg(
    R2_mean=("R2", "mean"), R2_std=("R2", "std"),
    RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
    MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
    CRPS_mean=("CRPS", "mean"), CRPS_std=("CRPS", "std"),
    WIS_mean=("WIS", "mean"), WIS_std=("WIS", "std"),
    PICP95_mean=("PICP95", "mean"), MPIW95_mean=("MPIW95", "mean"),
).sort_values("R2_mean", ascending=False)

cv_df.to_csv(os.path.join(OUTPUT_DIR, "group_cv_results.csv"), index=False)
cv_predictions_df.to_csv(os.path.join(OUTPUT_DIR, "group_cv_predictions.csv"), index=False)
selected_features_by_fold_df.to_csv(os.path.join(OUTPUT_DIR, "selected_features_by_fold.csv"), index=False)
cv_summary.to_csv(os.path.join(OUTPUT_DIR, "group_cv_summary.csv"), index=False)
print("\nGROUP-CV SUMMARY")
display(cv_summary.round(4))

# =============================================================================
# Permutation importance on the documented primary holdout
# =============================================================================
def predict_tabpfn_mean(X):
    return as_numpy(_tabpfn_full_output(tabpfn_model, X)["mean"]).reshape(-1)


def predict_dkl_mean(X):
    return predict_dkl_gp(
        dkl_model, dkl_likelihood, X, y_scaler, n_samples=50, seed=SEED + 3001
    ).mean


def predict_diff_mean(X):
    return predict_conditional_diffusion(
        diff_model, X, y_scaler, n_samples=100, seed=SEED + 3002
    ).mean


print("\nComputing exploratory hold-out permutation importance...")
pi_tab = custom_permutation_importance_rmse(
    predict_tabpfn_mean, X_test_tabpfn.copy(), y_test,
    X_train_tabpfn_df.columns.tolist(), n_repeats=20, random_state=SEED,
)
pi_dkl = custom_permutation_importance_rmse(
    predict_dkl_mean, X_test_sel.copy(), y_test,
    X_train_sel_df.columns.tolist(), n_repeats=20, random_state=SEED,
)
pi_diff = custom_permutation_importance_rmse(
    predict_diff_mean, X_test_sel.copy(), y_test,
    X_train_sel_df.columns.tolist(), n_repeats=20, random_state=SEED,
)
pi_tab.to_csv(os.path.join(OUTPUT_DIR, "importance_tabpfn.csv"), index=False)
pi_dkl.to_csv(os.path.join(OUTPUT_DIR, "importance_dkl.csv"), index=False)
pi_diff.to_csv(os.path.join(OUTPUT_DIR, "importance_diffusion.csv"), index=False)

# =============================================================================
# Plot helpers
# =============================================================================
def finish_figure(fig, filename_base):
    fig.tight_layout()
    save_figure(fig, filename_base)
    plt.show()
    plt.close(fig)


def compact_metric_figure(data, metrics, filename, title):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.4))
    axes = np.atleast_1d(axes)
    for ax, (column, label, higher_better) in zip(axes, metrics):
        ordered = data.sort_values(column, ascending=not higher_better)
        ax.bar(ordered["Model"], ordered[column])
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=18)
        for i, value in enumerate(ordered[column]):
            ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(title, y=1.02)
    finish_figure(fig, filename)


# =============================================================================
# 1) Per-model TEST parity plots with model-native 95% intervals
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), sharex=True, sharey=True)
for ax, model_name in zip(axes, ["TabPFN", "DKL_GP", "Diffusion"]):
    bundle = test_predictions[model_name]
    lower = bundle.quantile(0.025)
    upper = bundle.quantile(0.975)
    ax.errorbar(
        y_test, bundle.mean,
        yerr=np.vstack([bundle.mean - lower, upper - bundle.mean]),
        fmt=model_markers[model_name], ms=5,
        color=model_colors[model_name], ecolor=model_colors[model_name],
        elinewidth=0.8, capsize=2, alpha=0.82,
    )
    mn = min(y_test.min(), bundle.mean.min())
    mx = max(y_test.max(), bundle.mean.max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=1.2)
    metric = regression_metrics(y_test, bundle, gaussian_predictive=(model_name == "DKL_GP"))
    ax.set_title(
        f"{model_display[model_name]}\n"
        f"$R^2$={metric['R2']:.3f}, CRPS={metric['CRPS']:.2f}, PICP$_{{95}}$={metric['PICP95']:.2f}"
    )
    ax.set_xlabel("Actual burst pressure")
axes[0].set_ylabel("Predicted burst pressure")
fig.suptitle("Test parity plots with model-native 95% predictive intervals", y=1.03)
finish_figure(fig, "fig_01_test_parity_by_model")

# =============================================================================
# 2) Per-model TRAIN parity plots
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), sharex=True, sharey=True)
for ax, model_name in zip(axes, ["TabPFN", "DKL_GP", "Diffusion"]):
    bundle = train_predictions[model_name]
    lower = bundle.quantile(0.025)
    upper = bundle.quantile(0.975)
    ax.errorbar(
        y_train, bundle.mean,
        yerr=np.vstack([bundle.mean - lower, upper - bundle.mean]),
        fmt=model_markers[model_name], ms=4.3,
        color=model_colors[model_name], ecolor=model_colors[model_name],
        elinewidth=0.6, capsize=1.5, alpha=0.72,
    )
    mn = min(y_train.min(), bundle.mean.min())
    mx = max(y_train.max(), bundle.mean.max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=1.2)
    ax.set_title(f"{model_display[model_name]}\n$R^2$={r2_score(y_train, bundle.mean):.3f}")
    ax.set_xlabel("Actual burst pressure")
axes[0].set_ylabel("Predicted burst pressure")
fig.suptitle("Training parity plots with model-native 95% predictive intervals", y=1.03)
finish_figure(fig, "fig_02_train_parity_by_model")

# =============================================================================
# 3) All TEST models in one combined plot with +/-5% band
# =============================================================================
fig, ax = plt.subplots(figsize=(8.5, 7.2))
for model_name in ["TabPFN", "DKL_GP", "Diffusion"]:
    bundle = test_predictions[model_name]
    ax.scatter(
        y_test, bundle.mean, s=65, alpha=0.78,
        color=model_colors[model_name], marker=model_markers[model_name],
        label=f"{model_display[model_name]} ($R^2$={r2_score(y_test, bundle.mean):.3f})",
    )
mn = min([y_test.min()] + [b.mean.min() for b in test_predictions.values()])
mx = max([y_test.max()] + [b.mean.max() for b in test_predictions.values()])
ax.plot([mn, mx], [mn, mx], "k--", lw=1.3, label="Ideal")
ax.fill_between([mn, mx], [0.95 * mn, 0.95 * mx], [1.05 * mn, 1.05 * mx], color="gray", alpha=0.12, label="+/-5% error band")
ax.set_xlabel("Actual burst pressure")
ax.set_ylabel("Predicted burst pressure")
ax.set_title("Combined test-set comparison")
ax.legend(frameon=True, loc="best")
finish_figure(fig, "fig_03_combined_test_with_5pct_band")

# =============================================================================
# 4) All TRAIN models in one combined plot with +/-5% band
# =============================================================================
fig, ax = plt.subplots(figsize=(8.5, 7.2))
for model_name in ["TabPFN", "DKL_GP", "Diffusion"]:
    bundle = train_predictions[model_name]
    ax.scatter(
        y_train, bundle.mean, s=50, alpha=0.60,
        color=model_colors[model_name], marker=model_markers[model_name],
        label=f"{model_display[model_name]} ($R^2$={r2_score(y_train, bundle.mean):.3f})",
    )
mn = min([y_train.min()] + [b.mean.min() for b in train_predictions.values()])
mx = max([y_train.max()] + [b.mean.max() for b in train_predictions.values()])
ax.plot([mn, mx], [mn, mx], "k--", lw=1.3, label="Ideal")
ax.fill_between([mn, mx], [0.95 * mn, 0.95 * mx], [1.05 * mn, 1.05 * mx], color="gray", alpha=0.12, label="+/-5% error band")
ax.set_xlabel("Actual burst pressure")
ax.set_ylabel("Predicted burst pressure")
ax.set_title("Combined training-set comparison")
ax.legend(frameon=True, loc="best")
finish_figure(fig, "fig_04_combined_train_with_5pct_band")

# =============================================================================
# 5) Visual summary instead of a dense table
# =============================================================================
compact_metric_figure(
    summary_df,
    [("Test_R2", "Test $R^2$", True), ("Test_RMSE", "Test RMSE", False), ("Test_CRPS", "Test CRPS", False)],
    "fig_05_test_summary_bars",
    "Primary hold-out comparison",
)

# =============================================================================
# 6) Group-CV distributions
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, metric, label in zip(axes, ["R2", "RMSE", "CRPS"], ["Group-CV $R^2$", "Group-CV RMSE", "Group-CV CRPS"]):
    sns.boxplot(data=cv_df, x="Model", y=metric, hue="Model", legend=False, ax=ax)
    sns.stripplot(data=cv_df, x="Model", y=metric, color="black", alpha=0.50, size=4, ax=ax)
    ax.set_title(label)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=18)
finish_figure(fig, "fig_06_group_cv_boxplots")

# =============================================================================
# 7) Feature importance plots
# =============================================================================
importance_map = {"TabPFN": pi_tab, "DKL_GP": pi_dkl, "Diffusion": pi_diff}
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
for ax, model_name in zip(axes, ["TabPFN", "DKL_GP", "Diffusion"]):
    imp_df = importance_map[model_name].head(10).iloc[::-1]
    ax.barh(imp_df["Feature"], imp_df["Importance_Mean"], xerr=imp_df["Importance_STD"], color=model_colors[model_name], alpha=0.85)
    ax.set_title(f"{model_display[model_name]} feature importance")
    ax.set_xlabel("Increase in RMSE after permutation")
finish_figure(fig, "fig_07_feature_importance_by_model")

all_top = sorted(set().union(*[set(frame.head(10)["Feature"]) for frame in importance_map.values()]))
heat_df = pd.DataFrame(index=all_top)
for model_name, frame in importance_map.items():
    heat_df[model_display[model_name]] = frame.set_index("Feature").reindex(all_top)["Importance_Mean"]
heat_df = heat_df.fillna(0.0)
fig, ax = plt.subplots(figsize=(8, max(5, 0.35 * len(all_top))))
sns.heatmap(heat_df.sort_values(by="TabPFN", ascending=False), cmap="mako", annot=True, fmt=".3f", ax=ax)
ax.set_title("Cross-model permutation importance")
finish_figure(fig, "fig_08_feature_importance_heatmap")

# =============================================================================
# 8) Sorted predictive intervals for all probabilistic models
# =============================================================================
order_test = np.argsort(y_test)
y_test_sorted = y_test[order_test]
fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
for ax, model_name in zip(axes, ["TabPFN", "DKL_GP", "Diffusion"]):
    bundle = test_predictions[model_name]
    m = bundle.mean[order_test]
    lo = bundle.quantile(0.025)[order_test]
    hi = bundle.quantile(0.975)[order_test]
    ax.plot(y_test_sorted, "ko", ms=3.8, label="True")
    ax.plot(m, color=model_colors[model_name], label="Predicted mean")
    ax.fill_between(np.arange(len(m)), lo, hi, color=model_colors[model_name], alpha=0.22, label="Native/empirical 95% interval")
    ax.set_title(f"{model_display[model_name]} uncertainty")
    ax.set_xlabel("Sorted test samples")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Burst pressure")
finish_figure(fig, "fig_09_uncertainty_sorted_probabilistic")

# =============================================================================
# 9) Calibration versus sharpness (visual, not a dense table)
# =============================================================================
test_cal = calibration_df[calibration_df["Split"] == "Test"].copy()
fig, ax = plt.subplots(figsize=(7.5, 5.8))
for model_name in test_cal["Model"].unique():
    sub = test_cal[test_cal["Model"] == model_name]
    key = next(k for k, v in model_display.items() if v == model_name)
    ax.plot(sub["Mean_Interval_Width"], sub["Empirical_Coverage"], marker=model_markers[key], color=model_colors[key], label=model_name)
    for _, row in sub.iterrows():
        ax.annotate(f"{int(row['Nominal_Coverage']*100)}%", (row["Mean_Interval_Width"], row["Empirical_Coverage"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
ax.axhline(0.95, color="k", linestyle="--", linewidth=1.0, alpha=0.7)
ax.set_xlabel("Mean predictive interval width")
ax.set_ylabel("Empirical coverage")
ax.set_title("Calibration versus sharpness across interval levels")
ax.legend()
finish_figure(fig, "fig_10_calibration_sharpness_probabilistic")

# =============================================================================
# 10) Residual diagnostics
# =============================================================================
resid_rows = []
for model_name, bundle in test_predictions.items():
    for residual in y_test - bundle.mean:
        resid_rows.append({"Model": model_display[model_name], "Residual": residual})
resid_df = pd.DataFrame(resid_rows)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
sns.violinplot(data=resid_df, x="Model", y="Residual", hue="Model", legend=False, inner="box", cut=0, ax=axes[0])
axes[0].axhline(0, color="k", linestyle="--", linewidth=1.1)
axes[0].set_title("Residual distributions")
for model_name, bundle in test_predictions.items():
    abs_err = np.abs(y_test - bundle.mean)
    axes[1].scatter(
        df.iloc[test_idx]["Corrosion_Ratio"].values, abs_err,
        color=model_colors[model_name], marker=model_markers[model_name],
        alpha=0.65, label=model_display[model_name], s=45,
    )
axes[1].set_title("Absolute error vs corrosion ratio")
axes[1].set_xlabel("Corrosion ratio")
axes[1].set_ylabel("Absolute error")
axes[1].legend(frameon=True)
finish_figure(fig, "fig_11_residual_diagnostics")

# =============================================================================
# 11) Train-test generalization gap
# =============================================================================
gap_df = summary_df[["Model", "Train_R2", "Test_R2"]].copy()
gap_df["Gap_R2"] = gap_df["Train_R2"] - gap_df["Test_R2"]
fig, ax = plt.subplots(figsize=(7, 4.8))
ax.bar(gap_df["Model"], gap_df["Gap_R2"])
for i, value in enumerate(gap_df["Gap_R2"]):
    ax.text(i, value, f"{value:.3f}", ha="center", va="bottom")
ax.set_title("Train-test generalization gap")
ax.set_ylabel("Train $R^2$ - Test $R^2$")
ax.tick_params(axis="x", rotation=18)
finish_figure(fig, "fig_12_generalization_gap")

# =============================================================================
# 12) Split/source composition as a figure
# =============================================================================
split_composition = pd.DataFrame({
    "Publication_Source": sorted(df["Publication_Source"].unique()),
})
split_composition["Train"] = split_composition["Publication_Source"].map(df.iloc[train_idx]["Publication_Source"].value_counts()).fillna(0).astype(int)
split_composition["Test"] = split_composition["Publication_Source"].map(df.iloc[test_idx]["Publication_Source"].value_counts()).fillna(0).astype(int)
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(split_composition["Publication_Source"], split_composition["Train"], label="Train")
ax.bar(split_composition["Publication_Source"], split_composition["Test"], bottom=split_composition["Train"], label="Test")
ax.set_xlabel("Publication source")
ax.set_ylabel("Number of observations")
ax.set_title("Primary 80/20 holdout composition by publication source")
ax.legend()
finish_figure(fig, "fig_13_split_source_composition")

# Compact summary-table figure retained only because exact numerical values are useful.
summary_cols = ["Model", "Test_R2", "Test_RMSE", "Test_MAE", "Test_CRPS", "Test_WIS", "Test_PICP95", "Test_MPIW95"]
summary_for_figure = summary_df[summary_cols].copy().round(3)
fig, ax = plt.subplots(figsize=(12, 2.2 + 0.45 * len(summary_for_figure)))
ax.axis("off")
table = ax.table(cellText=summary_for_figure.values, colLabels=summary_for_figure.columns, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.4)
ax.set_title("Primary hold-out numerical summary", pad=14)
finish_figure(fig, "fig_14_holdout_summary_table")

# Physical-validity diagnostic figure
physical_test = physical_validity_df[physical_validity_df["Split"] == "Test"]
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.bar(physical_test["Model"], physical_test["Nonpositive_Sample_Fraction"])
ax.set_ylabel("Fraction of predictive samples <= 0")
ax.set_title("Physical-validity diagnostic for test predictive distributions")
ax.tick_params(axis="x", rotation=18)
for i, value in enumerate(physical_test["Nonpositive_Sample_Fraction"]):
    ax.text(i, value, f"{value:.4f}", ha="center", va="bottom")
finish_figure(fig, "fig_15_physical_validity_diagnostic")


# =============================================================================
# 16) RSTRENG same-specimen engineering benchmark: parity panels
# =============================================================================
if RUN_RSTRENG_BENCHMARK and not rstreng_holdout_comparison_df.empty:
    rstreng_plot_models = [
        ("RSTRENG", "RSTRENG_Prediction_SMYSplus10ksi_MPa", "#555555", "s"),
        ("TabPFN", "TabPFN_Mean_MPa", model_colors["TabPFN"], model_markers["TabPFN"]),
        ("DKL-GP", "DKL_GP_Mean_MPa", model_colors["DKL_GP"], model_markers["DKL_GP"]),
        (
            "Conditional diffusion",
            "Conditional_Diffusion_Mean_MPa",
            model_colors["Diffusion"],
            model_markers["Diffusion"],
        ),
    ]

    truth = rstreng_holdout_comparison_df["Measured_Burst_Pressure_MPa"].to_numpy(dtype=float)
    all_predictions_for_limits = [
        rstreng_holdout_comparison_df[column].to_numpy(dtype=float)
        for _, column, _, _ in rstreng_plot_models
    ]
    global_min = min([truth.min()] + [arr.min() for arr in all_predictions_for_limits])
    global_max = max([truth.max()] + [arr.max() for arr in all_predictions_for_limits])
    pad = 0.06 * max(global_max - global_min, 1.0)
    plot_min = global_min - pad
    plot_max = global_max + pad

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 9.0), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (model_label, prediction_column, color, marker) in zip(
        axes, rstreng_plot_models
    ):
        prediction = rstreng_holdout_comparison_df[prediction_column].to_numpy(dtype=float)
        metrics_row = rstreng_same_subset_metrics_df.loc[
            rstreng_same_subset_metrics_df["Model"].eq(model_label)
        ].iloc[0]

        ax.scatter(
            truth,
            prediction,
            s=62,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.90,
        )
        ax.plot(
            [plot_min, plot_max],
            [plot_min, plot_max],
            "k--",
            linewidth=1.2,
            label="Ideal",
        )
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{model_label}\n"
            rf"$R^2$={metrics_row['R2']:.3f}, RMSE={metrics_row['RMSE']:.2f} MPa"
        )
        ax.set_xlabel("Measured burst pressure (MPa)")
        ax.set_ylabel("Predicted burst pressure (MPa)")
        ax.grid(True, alpha=0.18)

    fig.suptitle(
        "Same-specimen comparison on the RSTRENG-applicable hold-out subset",
        y=1.01,
    )
    finish_figure(fig, "fig_16_rstreng_same_subset_parity")


# =============================================================================
# 17) RSTRENG same-specimen benchmark: predicted/measured ratio by specimen
# =============================================================================
if RUN_RSTRENG_BENCHMARK and not rstreng_holdout_comparison_df.empty:
    fig, ax = plt.subplots(figsize=(11.2, 5.4))

    specimen_labels = (
        rstreng_holdout_comparison_df["ID"]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .tolist()
    )
    x = np.arange(len(specimen_labels), dtype=float)
    offsets = [-0.27, -0.09, 0.09, 0.27]

    ratio_plot_models = [
        ("RSTRENG", "RSTRENG_Prediction_SMYSplus10ksi_MPa", "#555555", "s"),
        ("TabPFN", "TabPFN_Mean_MPa", model_colors["TabPFN"], model_markers["TabPFN"]),
        ("DKL-GP", "DKL_GP_Mean_MPa", model_colors["DKL_GP"], model_markers["DKL_GP"]),
        (
            "Conditional diffusion",
            "Conditional_Diffusion_Mean_MPa",
            model_colors["Diffusion"],
            model_markers["Diffusion"],
        ),
    ]

    truth = rstreng_holdout_comparison_df["Measured_Burst_Pressure_MPa"].to_numpy(dtype=float)

    for offset, (model_label, prediction_column, color, marker) in zip(
        offsets, ratio_plot_models
    ):
        prediction = rstreng_holdout_comparison_df[prediction_column].to_numpy(dtype=float)
        ratio = prediction / truth
        ax.scatter(
            x + offset,
            ratio,
            s=58,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
            label=model_label,
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Perfect agreement")
    ax.set_xticks(x)
    ax.set_xticklabels(specimen_labels, rotation=30, ha="right")
    ax.set_ylabel("Predicted / measured burst pressure")
    ax.set_xlabel("Hold-out specimen")
    ax.set_title(
        "Prediction ratio on the RSTRENG-applicable hold-out subset"
    )
    ax.legend(frameon=True, ncol=3)
    ax.grid(True, axis="y", alpha=0.18)
    finish_figure(fig, "fig_17_rstreng_same_subset_prediction_ratio")


# =============================================================================
# 18) RSTRENG same-specimen benchmark: compact metric comparison
# =============================================================================
if RUN_RSTRENG_BENCHMARK and not rstreng_same_subset_metrics_df.empty:
    ordered_models = [
        "RSTRENG",
        "TabPFN",
        "DKL-GP",
        "Conditional diffusion",
    ]
    metric_plot_df = (
        rstreng_same_subset_metrics_df
        .set_index("Model")
        .loc[ordered_models]
        .reset_index()
    )

    bar_colors = [
        "#555555",
        model_colors["TabPFN"],
        model_colors["DKL_GP"],
        model_colors["Diffusion"],
    ]
    display_labels = ["RSTRENG", "TabPFN", "DKL-GP", "Conditional\ndiffusion"]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    metric_specs = [
        ("R2", r"$R^2$", True),
        ("RMSE", "RMSE (MPa)", False),
        ("MAE", "MAE (MPa)", False),
    ]

    for ax, (column, title, higher_is_better) in zip(axes, metric_specs):
        values = metric_plot_df[column].to_numpy(dtype=float)
        bars = ax.bar(display_labels, values, color=bar_colors, alpha=0.90)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=12)
        ax.grid(True, axis="y", alpha=0.18)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        if column == "R2":
            ax.set_ylim(min(0.0, np.nanmin(values) - 0.08), 1.02)

    fig.suptitle(
        "Point-prediction performance on the same RSTRENG-applicable hold-out subset",
        y=1.02,
    )
    finish_figure(fig, "fig_18_rstreng_same_subset_metrics")


# =============================================================================
# Reproducibility package
# =============================================================================
data_provenance_df = pd.DataFrame([
    ["Data file used", DATA_FILE],
    ["Source name", DATA_SOURCE_NAME],
    ["Citation", DATA_SOURCE_CITATION],
    ["Repository", DATA_SOURCE_URL],
    ["Rows in repository file used", len(df)],
    ["Selected original column positions", str(SELECTED_COLUMN_POSITIONS)],
], columns=["Field", "Value"])

feature_equations_df = pd.DataFrame([
    ["Corrosion_Ratio", "Corrosion_Depth / Wall_Thickness"],
    ["D_t_Ratio", "Outer_Diameter / Wall_Thickness"],
    ["Metal_Area", "pi*(OD/2)^2 - pi*(OD/2-WT)^2"],
    ["UTS_Wall_Interaction", "UTS * Wall_Thickness"],
    ["Corrosion_Area", "Corrosion_Length * Corrosion_Depth"],
    ["D_t_CorrosionRatio", "D_t_Ratio * Corrosion_Ratio"],
    ["Pressure_Factor", "UTS * Wall_Thickness / Outer_Diameter"],
    ["Corrosion_Severity", "Corrosion_Area / Metal_Area"],
    ["Thinness_Factor", "Wall_Thickness / Outer_Diameter"],
    ["Log_Corrosion_Length", "log(1 + Corrosion_Length)"],
    ["Log_Corrosion_Area", "log(1 + Corrosion_Area)"],
    ["Depth_sq", "Corrosion_Depth^2"],
    ["Length_sq", "Corrosion_Length^2"],
    ["Depth_Length", "Corrosion_Depth * Corrosion_Length"],
    ["Pressure_Severity", "Pressure_Factor * Corrosion_Severity"],
], columns=["Feature", "Exact_Equation"])

hyperparameters_df = pd.DataFrame([
    ["TabPFN", "n_estimators", TABPFN_N_ESTIMATORS],
    ["TabPFN", "expected package version", EXPECTED_TABPFN_VERSION],
    ["TabPFN", "actual package version", actual_tabpfn_version],
    ["TabPFN", "random_state", SEED],
    ["DKL-GP", "model seed", DKL_SEED],
    ["Diffusion", "model seed", DIFFUSION_SEED],
    ["TabPFN", "point prediction", "native predictive mean"],
    ["TabPFN", "intervals", "native criterion.icdf quantiles"],
    ["DKL-GP", "feature extractor", "input-64-32-8"],
    ["DKL-GP", "dropout", 0.05],
    ["DKL-GP", "kernel", "Matern nu=2.5, ARD=8"],
    ["DKL-GP", "likelihood", "GaussianLikelihood"],
    ["DKL-GP", "epochs holdout/CV", "250 / 180"],
    ["DKL-GP", "learning rate", 0.01],
    ["Diffusion", "type", "direct conditional target diffusion"],
    ["Diffusion", "T", 100],
    ["Diffusion", "beta schedule", "linear 1e-4 to 2e-2"],
    ["Diffusion", "hidden/time dimensions", "64 / 32"],
    ["Diffusion", "epochs holdout/CV", "700 / 450"],
    ["Diffusion", "learning rate", 0.002],
    ["Diffusion", "batch size", 16],
    ["All", "holdout predictive samples", N_PREDICTIVE_SAMPLES],
    ["All", "group-CV predictive samples", N_CV_PREDICTIVE_SAMPLES],
    ["RSTRENG", "benchmark enabled", RUN_RSTRENG_BENCHMARK],
    ["RSTRENG", "eligible geometry", "single longitudinal machined rectangular defects"],
    ["RSTRENG", "profile reconstruction", "uniform depth over reported corrosion length"],
    ["RSTRENG", "profile cells", RSTRENG_PROFILE_CELLS],
    ["RSTRENG", "primary flow stress", "SMYS + 10 ksi"],
], columns=["Model", "Hyperparameter", "Value"])

run_config_df = pd.DataFrame([
    ["Timestamp", datetime.now().isoformat(timespec="seconds")],
    ["Global/split seed", SEED],
    ["DKL model seed", DKL_SEED],
    ["Diffusion model seed", DIFFUSION_SEED],
    ["Device", DEVICE],
    ["Primary split", "Stratified 80/20 holdout (seed 123)"],
    ["Test fraction", 0.20],
    ["Split stratification", "Eight Split_Cluster strata (reproduces documented indices)"],
    ["Publication sources", 9],
    ["USE_IS_IRREGULAR", USE_IS_IRREGULAR],
    ["USE_FEATURE_SELECTION", USE_FEATURE_SELECTION],
    ["FS_MI_THRESHOLD", FS_MI_THRESHOLD],
    ["FS_MAX_FEATURES", FS_MAX_FEATURES],
    ["External scaling for TabPFN", True],
    ["Reason scaling retained", "Comparability with the reported study configuration"],
    ["RUN_RSTRENG_BENCHMARK", RUN_RSTRENG_BENCHMARK],
    ["RSTRENG_PROFILE_CELLS", RSTRENG_PROFILE_CELLS],
    ["RSTRENG primary reference stress", "SMYS by reported API 5L grade"],
], columns=["Setting", "Value"])

split_indices_df = pd.concat([
    pd.DataFrame({"Original_Index": train_idx, "Split": "Train"}),
    pd.DataFrame({"Original_Index": test_idx, "Split": "Test"}),
], ignore_index=True).sort_values("Original_Index")
split_indices_df["Publication_Source"] = split_indices_df["Original_Index"].map(df["Publication_Source"])
split_indices_df["Split_Cluster"] = split_indices_df["Original_Index"].map(df["Split_Cluster"])

source_definitions_df = pd.DataFrame([
    [1, 0, 7, 8],
    [2, 8, 27, 20],
    [3, 28, 42, 15],
    [4, 43, 50, 8],
    [5, 51, 55, 5],
    [6, 56, 61, 6],
    [7, 62, 62, 1],
    [8, 63, 98, 36],
    [9, 99, 103, 5],
], columns=[
    "Publication_Source",
    "Start_Index_Inclusive",
    "End_Index_Inclusive",
    "N"
])

engineered_audit_df = pd.DataFrame({
    "Original_Position_Zero_Based": [np.nan] * len(master_features),
    "Original_Column_Name": [""] * len(master_features),
    "Standardized_Name": master_features,
    "Status": ["ENGINEERED / MODEL INPUT"] * len(master_features),
    "Role_or_Reason": ["Model input after feature engineering or encoding"] * len(master_features),
})
columns_df = pd.concat([column_audit_df, engineered_audit_df], ignore_index=True)

software_rows = []
for package in ["python", "numpy", "pandas", "scipy", "scikit-learn", "torch", "gpytorch", "tabpfn", "matplotlib", "seaborn", "openpyxl", "joblib"]:
    if package == "python":
        version = platform.python_version()
    else:
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "not found"
    software_rows.append([package, version])
software_versions_df = pd.DataFrame(software_rows, columns=["Package", "Version"])

# Save models/scalers and exact split arrays.
model_dir = Path(OUTPUT_DIR) / "saved_models"
model_dir.mkdir(exist_ok=True)
torch.save({
    "model_state_dict": dkl_model.state_dict(),
    "likelihood_state_dict": dkl_likelihood.state_dict(),
    "selected_features": selected_features,
}, model_dir / "dkl_gp_state.pt")
torch.save({
    "model_state_dict": diff_model.state_dict(),
    "selected_features": selected_features,
}, model_dir / "conditional_diffusion_state.pt")
joblib.dump(x_scaler_tabpfn, model_dir / "x_scaler_tabpfn.joblib")
joblib.dump(x_scaler_sel, model_dir / "x_scaler_selected.joblib")
joblib.dump(y_scaler, model_dir / "y_scaler.joblib")
np.savez(model_dir / "split_indices.npz", train_idx=train_idx, test_idx=test_idx)

# Try the official TabPFN fitted-model saver when available.
try:
    from tabpfn.model_loading import save_fitted_tabpfn_model
    save_fitted_tabpfn_model(tabpfn_model, model_dir / "tabpfn_fitted.tabpfn_fit")
    tabpfn_save_status = "Saved with tabpfn.model_loading.save_fitted_tabpfn_model"
except Exception as exc:
    tabpfn_save_status = f"Not saved; reload and refit using stored split/scaler. Reason: {type(exc).__name__}: {exc}"

readme_text = f"""PROBABILISTIC BURST-PRESSURE PIPELINE
Generated: {datetime.now().isoformat(timespec='seconds')}

Primary analysis:
- Documented seed={SEED} and stratified 80/20 holdout logic.
- Study-specific model feature handling is retained for reproducibility.
- TabPFN uncertainty from its native predictive distribution.
- DKL-GP uncertainty from its Gaussian predictive distribution.
- Diffusion uncertainty from empirical generated samples.
- No Gaussian mean +/- 1.96*std assumption for TabPFN or diffusion.

Grouping detail:
- Split_Cluster retains the eight strata used to reproduce the documented holdout indices.
- Publication_Source records the corrected nine literature sources and is used for supplementary group-CV.

RSTRENG engineering benchmark:
- Restricted to single, longitudinal, machined/spark-eroded rectangular defects.
- Interacting/multiple, irregular/natural, intact, hoop, and angled defects are excluded.
- Rectangular profiles are reconstructed as uniform depth over the reported corrosion length.
- Primary RSTRENG flow stress is SMYS + 10 ksi.
- All four models are compared on the exact same RSTRENG-applicable hold-out specimens.

TabPFN persistence: {tabpfn_save_status}
"""
(Path(OUTPUT_DIR) / "README.txt").write_text(readme_text, encoding="utf-8")
shutil.copy2(DATA_FILE, Path(OUTPUT_DIR) / "original_Data.xlsx")
try:
    if "__file__" in globals() and Path(__file__).exists():
        shutil.copy2(__file__, Path(OUTPUT_DIR) / "exact_code_used.py")
except Exception:
    pass

# Create preliminary file manifest (the workbook itself is excluded to avoid self-hashing).
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

manifest_rows = []
for path in sorted(Path(OUTPUT_DIR).rglob("*")):
    if path.is_file() and path.name != "reproducibility_master.xlsx":
        manifest_rows.append({
            "Relative_Path": str(path.relative_to(OUTPUT_DIR)),
            "Size_Bytes": path.stat().st_size,
            "SHA256": sha256_file(path),
        })
manifest_df = pd.DataFrame(manifest_rows)

workbook_path = Path(OUTPUT_DIR) / "reproducibility_master.xlsx"
with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
    run_config_df.to_excel(writer, sheet_name="Run_Config", index=False)
    data_provenance_df.to_excel(writer, sheet_name="Data_Provenance", index=False)
    df.to_excel(writer, sheet_name="Cleaned_Dataset", index=False)
    cleaning_log_df.to_excel(writer, sheet_name="Cleaning_Log", index=False)
    columns_df.to_excel(writer, sheet_name="Columns_Used_Removed", index=False)
    feature_equations_df.to_excel(writer, sheet_name="Feature_Equations", index=False)
    source_definitions_df.to_excel(writer, sheet_name="Source_Definitions", index=False)
    split_indices_df.to_excel(writer, sheet_name="Split_Indices", index=False)
    split_composition.to_excel(writer, sheet_name="Split_Composition", index=False)
    pd.DataFrame({"Selected_Feature": selected_features}).to_excel(writer, sheet_name="Selected_Features", index=False)
    mi_scores.rename("MI_Score").rename_axis("Feature").reset_index().to_excel(writer, sheet_name="MI_Scores", index=False)
    hyperparameters_df.to_excel(writer, sheet_name="Hyperparameters", index=False)
    summary_df.to_excel(writer, sheet_name="Holdout_Metrics", index=False)
    pred_df.to_excel(writer, sheet_name="Holdout_Predictions", index=False)
    train_pred_df.to_excel(writer, sheet_name="Training_Predictions", index=False)
    calibration_df.to_excel(writer, sheet_name="Calibration", index=False)
    physical_validity_df.to_excel(writer, sheet_name="Physical_Validity", index=False)
    rstreng_benchmark_config_df.to_excel(writer, sheet_name="RSTRENG_Config", index=False)
    rstreng_eligibility_df.to_excel(writer, sheet_name="RSTRENG_Eligibility", index=False)
    rstreng_all_applicable_df.to_excel(writer, sheet_name="RSTRENG_All_Applicable", index=False)
    rstreng_holdout_comparison_df.to_excel(writer, sheet_name="RSTRENG_Holdout_Compare", index=False)
    rstreng_same_subset_metrics_df.to_excel(writer, sheet_name="RSTRENG_Holdout_Metrics", index=False)
    cv_df.to_excel(writer, sheet_name="Group_CV_Results", index=False)
    cv_summary.to_excel(writer, sheet_name="Group_CV_Summary", index=False)
    cv_predictions_df.to_excel(writer, sheet_name="Group_CV_Predictions", index=False)
    selected_features_by_fold_df.to_excel(writer, sheet_name="CV_Selected_Features", index=False)
    pi_tab.to_excel(writer, sheet_name="Importance_TabPFN", index=False)
    pi_dkl.to_excel(writer, sheet_name="Importance_DKL_GP", index=False)
    pi_diff.to_excel(writer, sheet_name="Importance_Diffusion", index=False)
    software_versions_df.to_excel(writer, sheet_name="Software_Versions", index=False)
    manifest_df.to_excel(writer, sheet_name="File_Manifest", index=False)

    workbook = writer.book
    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for column_cells in worksheet.columns:
            max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
            worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_length + 2, 10), 45)

print("\nREPRODUCIBILITY MASTER WORKBOOK CREATED")
print(workbook_path)
print("\nDATA PROVENANCE")
display(data_provenance_df)
print("\nSELECTED FEATURES")
display(pd.DataFrame({"Selected_Feature": selected_features}))
print("\nHYPERPARAMETERS")
display(hyperparameters_df)
print("\nSOFTWARE VERSIONS")
display(software_versions_df)

print("\nAll outputs saved to:")
print(OUTPUT_DIR)
print("Done.")