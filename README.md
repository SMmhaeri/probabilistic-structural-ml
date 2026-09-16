# Probabilistic Structural Machine Learning

Research code for **probabilistic failure-pressure prediction of corroded pipelines** using three distinct learning paradigms together with a mechanics-based engineering benchmark.

The project forms the machine-learning component of my M.Sc. research in Structural Engineering at the University of Alberta. Its emphasis is not only point-prediction accuracy, but also **predictive uncertainty, calibration, sharpness, reproducibility, and robustness to experimental-source shift**.

## Models

The research pipeline compares:

1. **TabPFN regression**
   - Uses the model's native predictive distribution.
   - Predictive intervals are extracted from native quantiles rather than imposing a Gaussian approximation.

2. **Deep-Kernel Gaussian Process (DKL-GP)**
   - Neural feature extractor coupled to a Gaussian-process model.
   - Predictive intervals are obtained from the Gaussian predictive distribution.

3. **Conditional diffusion regression**
   - Generates conditional predictive samples for failure pressure.
   - Intervals and quantiles are obtained empirically from generated samples.

4. **RSTRENG engineering benchmark**
   - Applied only to a defensible subset of single, longitudinal, machined/spark-eroded rectangular defects where a uniform-depth axial profile can be reconstructed without inventing unknown corrosion morphology.
   - Uses the standard `SMYS + 10 ksi` flow-stress definition for the primary comparison.

## Dataset

The pipeline uses the public burst-pressure compilation associated with:

> Cai, J., Jiang, X., Yang, Y., Lodewijks, G., and Wang, M. (2022). “Data-driven Methods to Predict the Burst Strength of Corroded Line Pipelines Subjected to Internal Pressure.” *Journal of Marine Science and Application*, 21(2), 115–132. DOI: 10.1007/s11804-022-00263-0.

Original public repository:

https://github.com/jiejie168/pipeBurstPressure_prediction_dataDriven

The spreadsheet used in this research contains **104 complete observations** from **nine literature-defined experimental source groups**. The analysis retains all 104 observations and fails explicitly if required values are missing or invalid rather than silently changing the research sample.

The raw variables selected by the current pipeline are:

- measured failure pressure;
- ultimate tensile strength (UTS);
- outside diameter;
- wall thickness;
- corrosion length;
- corrosion depth;
- notch/defect shape, used to derive an irregular-defect indicator.

## Feature engineering

The workflow constructs physically interpretable features including:

```text
Corrosion_Ratio         = Corrosion_Depth / Wall_Thickness
D_t_Ratio               = Outer_Diameter / Wall_Thickness
Metal_Area              = π(OD/2)^2 - π(OD/2 - WT)^2
UTS_Wall_Interaction    = UTS × Wall_Thickness
Corrosion_Area          = Corrosion_Length × Corrosion_Depth
D_t_CorrosionRatio      = D_t_Ratio × Corrosion_Ratio
Pressure_Factor         = UTS × Wall_Thickness / Outer_Diameter
Corrosion_Severity      = Corrosion_Area / Metal_Area
Thinness_Factor         = Wall_Thickness / Outer_Diameter
Log_Corrosion_Length    = log(1 + Corrosion_Length)
Log_Corrosion_Area      = log(1 + Corrosion_Area)
Depth_sq                = Corrosion_Depth²
Length_sq               = Corrosion_Length²
Depth_Length            = Corrosion_Depth × Corrosion_Length
Pressure_Severity       = Pressure_Factor × Corrosion_Severity
```

The code can optionally remove the irregular-defect indicator and can optionally apply training-only mutual-information feature selection for DKL-GP and diffusion. TabPFN retains the full active feature set.

## Validation strategy

### Primary holdout

The primary comparison preserves the documented stratified **80/20 split with seed 123**. The code contains a regression guard for the expected 21 holdout indices so an accidental change in the split cannot silently be reported as the same analysis.

### Source-aware validation

A supplementary **grouped cross-validation** analysis uses the corrected literature-source labels to evaluate performance when entire experimental sources are held out. This probes uncertainty robustness under source/domain shift rather than relying only on a mixed-source random holdout.

## Probabilistic evaluation

The workflow evaluates both conventional regression and distributional metrics, including:

- R², RMSE, and MAE;
- Continuous Ranked Probability Score (**CRPS**);
- Weighted Interval Score (**WIS**);
- prediction-interval coverage probability (**PICP**);
- mean prediction-interval width (**MPIW**);
- calibration across nominal 50%, 80%, 90%, and 95% intervals;
- residual/error characteristics;
- cross-model permutation importance.

For TabPFN and diffusion, the code does **not** impose `mean ± 1.96 × standard deviation` to construct 95% intervals; it uses model-native or empirical predictive quantiles.

## Reproducibility

The research script fixes and records random seeds, model settings, software versions, split indices, feature definitions, predictions, calibration diagnostics, and file hashes.

Important defaults include:

```text
Global seed                    : 123
TabPFN estimators              : 8
Expected TabPFN version        : 8.1.0
Final predictive samples       : 1000
Group-CV predictive samples    : 300
DKL-GP holdout/CV epochs       : 250 / 180
DKL-GP learning rate           : 0.01
Diffusion holdout/CV epochs    : 700 / 450
Diffusion timesteps            : 100
Diffusion learning rate        : 0.002
```

The pipeline writes a `reproducibility_master.xlsx` workbook containing run configuration, data provenance, cleaned data, column audit, exact feature equations, source definitions, split indices, feature-selection records, hyperparameters, model metrics, predictions, calibration results, RSTRENG eligibility and comparison tables, group-CV results, feature importance, software versions, and a file manifest with SHA-256 hashes.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── CITATION.cff
├── data/
│   └── README.md
└── src/
    └── probabilistic_burst_pressure.py
```

## Installation

A GPU-enabled environment is recommended for the DKL-GP and diffusion components.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The reference research environment used **TabPFN 8.1.0**. TabPFN may require its own model-weight download and license-acceptance step depending on the installed version and execution environment.

## Running

Download the public source spreadsheet from the dataset repository above and place the required workbook in the working directory as:

```text
Data.xlsx
```

Then run:

```bash
python src/probabilistic_burst_pressure.py
```

Outputs are written to a run-specific directory whose name records the seed and feature-switch settings.

## Related thesis

> **Seyyed Mohammad Mojtahed Haeri.** *Uncertainty Propagation in Burst-Pressure Assessment of Corroded Pipelines using Engineering and Machine Learning Methods.* M.Sc. thesis, University of Alberta, 2026.

## Data and scope

This repository is built around a **public literature dataset**. Proprietary industry inspection data and other restricted materials are outside the repository.

The results are research outputs and should not be interpreted as a substitute for engineering codes, standards, or professional integrity-management judgment.

## License

A software license has **not yet been assigned** while publication and IP permissions are being finalized. Publication of the repository does not by itself grant reuse rights beyond those provided by applicable law and GitHub's terms.
