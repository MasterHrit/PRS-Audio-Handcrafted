
# PRS — Handcrafted Audio Feature Extraction and Classification

**Project:** Audio-Visual Emotion Recognition through Hybrid Feature Analysis and Multimodal Fusion  
**Programme:** NUS MTech in Artificial Intelligence Systems  
**Module:** Pattern Recognition Systems (PRS)  
**Contribution:** Member 1 (M1) — Handcrafted Audio Features and Audio Feature Fusion

## 1. Overview

This repository implements handcrafted audio feature extraction and classification for speaker-independent emotion recognition using the RAVDESS speech dataset.

The work includes:

- Audio preprocessing and validation
- Time-domain, spectral, and wavelet feature extraction
- Comparison of 30 preprocessing configurations
- Evaluation of nine machine learning classifiers
- Hyperparameter tuning of five shortlisted classifiers
- Preparation for fusion with M2's deep-learned audio features

The repository maintains two implementations:

- **Baseline V1:** Original 48D handcrafted feature pipeline.
- **Experimental V2:** Configurable preprocessing and feature extraction with systematic classifier comparisons.

## 2. Dataset

**Dataset:** RAVDESS — Audio-only speech subset  
**Source:** https://zenodo.org/records/1188976

| Property | Value |
|---|---|
| Recordings | 1,440 |
| Actors | 24 |
| Emotion classes | 8 |
| Sampling rate | 16 kHz |
| Feature dimensions | 48 |

Emotion classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised.

### Actor-Independent Splitting

| Partition | Actors | Recordings |
|---|---:|---:|
| Training | 16 | 960 |
| Validation | 4 | 240 |
| Testing | 4 | 240 |

Recordings from the same actor are not shared across partitions.

The raw RAVDESS recordings are not included in the repository.

## 3. Handcrafted Feature Representation

| Feature Group | Description | Dimensions |
|---|---|---:|
| Time-domain | ZCR, RMS, Energy | 12 |
| Spectral | FFT-based spectral descriptors | 16 |
| Wavelet | DWT coefficient statistics | 20 |
| **Combined** | **Concatenated handcrafted features** | **48** |

Experimental V2 evaluates:

- **Wavelets:** Haar, db2, db4, sym4, db8
- **Normalization:** Raw and normalized
- **Frame lengths:** 20, 25, 30 ms

This produces **30 combined 48D datasets**.

## 4. Repository Structure

```text
M1_Audio_Handcrafted/
├── data/
│   ├── raw/
│   ├── metadata.csv
│   ├── actor_splits.csv
│   └── audio_splits.csv
│
├── src/
│   ├── dataset.py
│   ├── split_dataset.py
│   ├── preprocessing.py
│   ├── baseline_v1/
│   ├── experimental_v2/
│   └── experiments/
│       ├── compare_preprocessing.py
│       ├── tuning_utils.py
│       ├── tune_logistic_regression.py
│       ├── tune_mlp.py
│       ├── tune_svc.py
│       ├── tune_xgboost.py
│       └── tune_random_forest.py
│
├── notebooks/
├── docs/
├── outputs/
│   ├── features/
│   │   ├── baseline_v1/
│   │   └── experimental_v2/
│   ├── metrics/
│   └── figures/
│
├── pyproject.toml
├── uv.lock
└── README.md
```

Generated outputs and raw recordings may be excluded from version control.

## 5. Installation

**Requirements:** Python 3.11 and uv.

From the repository root:

```bash
uv sync --locked
```

Install the locked dependencies before running the pipeline.

## 6. Pipeline Execution

### Dataset Preparation

```bash
uv run --locked python -m src.dataset
uv run --locked python -m src.split_dataset
```

### Baseline V1

```bash
uv run --locked python -m src.baseline_v1.feature_pipeline
```

### Experimental V2

Run sequentially:

```bash
uv run --locked python -m src.experimental_v2.run_preprocessing_validation
uv run --locked python -m src.experimental_v2.extract_time_dataset
uv run --locked python -m src.experimental_v2.extract_spectral_dataset
uv run --locked python -m src.experimental_v2.extract_wavelet_dataset
uv run --locked python -m src.experimental_v2.combine_features
```

### Classifier Comparison

```bash
uv run --locked python -m src.experiments.compare_preprocessing
```

### Hyperparameter Tuning

```bash
uv run --locked python -m src.experiments.tune_logistic_regression
uv run --locked python -m src.experiments.tune_mlp
uv run --locked python -m src.experiments.tune_svc
uv run --locked python -m src.experiments.tune_xgboost
uv run --locked python -m src.experiments.tune_random_forest
```

## 7. Experimental Results

### Initial Classifier Comparison

Nine classifiers were evaluated across 30 preprocessing configurations, producing **270 experiments**.

The five highest-performing classifiers were shortlisted for tuning.

### Default vs Tuned Validation Macro-F1

| Classifier | Default | Tuned |
|---|---:|---:|
| Logistic Regression | **0.3885** | 0.3745 |
| MLP | **0.3692** | 0.3331 |
| XGBoost | **0.3687** | 0.3627 |
| SVC | 0.3413 | **0.3535** |
| Random Forest | 0.3259 | **0.3315** |

### Current Leading Configuration

| Setting | Value |
|---|---|
| Classifier | Logistic Regression (default) |
| Wavelet | db2 |
| Normalization | Normalized |
| Frame length | 20 ms |
| Validation accuracy | 40.00% |
| Validation Macro-F1 | 0.3885 |

This is the highest observed validation Macro-F1 from the current handcrafted classifier comparison.

The validation results are exploratory because the same validation partition was used during preprocessing selection. Final model evaluation remains pending.

## 8. Documentation

Detailed methodology and experimental findings are available in:

| Document | Description |
|---|---|
| [Handcrafted Feature Pipeline](docs/01_handcrafted_feature_pipeline.md) | Dataset, preprocessing, feature extraction, and execution |
| [Preprocessing and Classifier Comparison](docs/02_preprocessing_and_classifier_comparison.md) | Initial 270 experiments and classifier shortlist |
| [Hyperparameter Tuning Results](docs/03_hyperparameter_tuning_results.md) | Five-model tuning and default-versus-tuned evaluation |

The `notebooks/` directory contains supporting exploratory analysis, comparisons, and visualizations.

## 9. Current Status

| Component | Status |
|---|---|
| Dataset preparation and actor splitting | Complete |
| Baseline V1 feature extraction | Complete |
| Experimental V2 feature extraction | Complete |
| Preprocessing validation | Complete |
| 30 preprocessing configurations | Complete |
| Nine-classifier comparison | Complete |
| Five-classifier hyperparameter tuning | Complete |
| Comparative documentation | Complete |
| Feature ablation | Planned |
| M1 + M2 deep-feature comparison | Planned |
| M5 fusion handover | Planned |

## 10. Next Steps

1. Finalize the handcrafted feature representation and classifier.
2. Perform feature ablation experiments.
3. Align M1's 48D handcrafted features with M2's 128D deep-learned representations.
4. Compare handcrafted-only, deep-only, and concatenated audio representations.
5. Prepare aligned features and classifier probabilities for M5's multimodal fusion experiments.

The held-out test partition should remain untouched until the final evaluation protocol is established.
