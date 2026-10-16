# PRS — Audio Handcrafted Feature Extraction

**Project:** Multimodal Emotion Recognition Using Speech and Facial Expressions  
**Programme:** NUS MTech in Artificial Intelligence Systems  
**Module:** Pattern Recognition Systems (PRS)  
**Component:** Member 1 — Audio Handcrafted Feature Extraction and Audio Feature Fusion

## 1. Overview

This repository contains the handcrafted audio feature extraction and classification component of the PRS group project.

The objective is to investigate how time-domain, frequency-domain, and time-frequency signal-processing features contribute to eight-class speech emotion recognition using the RAVDESS dataset.

The implementation also supports subsequent feature-level fusion with deep-learned audio representations developed by Member 2.

## 2. Dataset

**Dataset:** Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)

**Source:** https://zenodo.org/records/1188976

Only the audio-only speech subset is used in this component.

- 1,440 speech recordings
- 24 actors
- 8 emotion classes: neutral, calm, happy, sad, angry, fearful, disgust, surprise
- Original sampling rate: 48 kHz
- Preprocessing sampling rate: 16 kHz

The original audio recordings are stored locally under `data/raw/` and are excluded from Git version control.

## 3. Environment Setup

The project uses Python 3.11 and `uv` for dependency management.

Install dependencies:

```bash
uv sync --locked
```

Run scripts:

```bash
uv run --locked python src/dataset.py
uv run --locked python src/split_dataset.py
uv run --locked python src/preprocessing.py
uv run --locked python src/time_features.py
uv run --locked python src/spectral_features.py
uv run --locked python src/feature_pipeline.py
```

Dependencies are managed through `pyproject.toml` and `uv.lock`.

## 4. Data Preparation

### Dataset Indexing

Implemented in `src/dataset.py`.

- Parses RAVDESS filenames to extract actor IDs, emotion labels, intensity, statement and repetition.
- Generates `data/metadata.csv`.
- Validates 1,440 recordings across 24 actors and eight emotions.

### Actor-Independent Partitioning

Implemented in `src/split_dataset.py`.

A reproducible gender-balanced split is generated using random seed 42.

| Partition | Actors | Recordings |
|---|---:|---:|
| Training | 16 | 960 |
| Validation | 4 | 240 |
| Testing | 4 | 240 |

Each actor belongs exclusively to one partition.

The shared actor assignments are saved in `data/actor_splits.csv`. Audio-specific sample assignments are saved in `data/audio_splits.csv`.

## 5. Audio Preprocessing

Implemented in `src/preprocessing.py`.

The preprocessing pipeline includes:

1. Loading mono audio.
2. Resampling from 48 kHz to 16 kHz.
3. Trimming leading and trailing silence using `top_db=35`.
4. Optional peak amplitude normalization.

Peak normalization is disabled for the initial handcrafted baseline to preserve amplitude-related information for RMS and Energy analysis.

### Preprocessing Validation

Automated validation was performed on all 1,440 recordings.

| Metric | Result |
|---|---:|
| Successfully processed | 1,440 |
| Processing failures | 0 |
| Mean original duration | 3.70 seconds |
| Mean processed duration | 2.01 seconds |
| Mean retained duration | 53.95% |
| Minimum retained duration | 38.56% |

The preprocessing validation results are saved in `outputs/preprocessing_validation.csv`.

## 6. Handcrafted Feature Extraction

### Time-Domain Features

Implemented in `src/time_features.py`.

- Zero-Crossing Rate (ZCR)
- Root Mean Square (RMS)
- Frame Energy

### Frequency-Domain Features

Implemented in `src/spectral_features.py`.

- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff (85% cumulative spectral power)
- Spectral Energy

### Framing Configuration

| Parameter | Value |
|---|---|
| Sampling rate | 16 kHz |
| Frame length | 400 samples (25 ms) |
| Hop length | 160 samples (10 ms) |
| Overlap | 60% |
| Spectral window | Hann |

### Feature Aggregation

Implemented in `src/feature_pipeline.py`.

Each frame-level feature is summarized using:

- Mean
- Standard deviation
- Minimum
- Maximum

This produces 28 fixed-length features per recording:

- 12 time-domain features
- 16 spectral features

**Generated feature matrix:** 1,440 × 28

Saved to `outputs/features/time_spectral_features.csv`.

### Feature Validation

- All 1,440 recordings processed successfully.
- No missing or non-finite feature values detected.
- No duplicate sample IDs.
- All 28 feature columns have nonzero variance.
- Training/validation/test sample counts verified.

## Wavelet Feature Extraction

Discrete Wavelet Transform (DWT) features are extracted using the Daubechies-4 (`db4`) wavelet with four decomposition levels.

Five coefficient groups are analyzed: A4, D4, D3, D2 and D1.

For each group, mean, variance, mean squared energy and Shannon entropy are calculated, producing 20 DWT features.

The complete handcrafted representation contains:

| Feature group | Dimensions |
|---|---:|
| Time-domain | 12 |
| Spectral | 16 |
| DWT | 20 |
| **Total** | **48** |

Training-set correlation analysis identified near-perfect correlations between DWT variance and mean squared energy within each coefficient group. A reduced 43-dimensional representation was therefore included in the classification experiments.

## Handcrafted Classification

Five handcrafted feature configurations were evaluated using SVM and MLP classifiers on actor-independent training and validation partitions.

| Representation | Best classifier | Validation Macro-F1 |
|---|---|---:|
| Time-domain (12D) | SVM | 0.3334 |
| Spectral (16D) | SVM | 0.2699 |
| DWT (20D) | MLP | 0.2429 |
| Full handcrafted (48D) | MLP | **0.3518** |
| Reduced handcrafted (43D) | SVM | 0.3408 |

Feature scaling was fitted on the training partition only.

## Deep-Learned Audio Feature Integration

Member 2 supplied 128-dimensional pooled representations from a Transformer-based audio encoder.

All 1,440 recordings were matched between the handcrafted and deep-learned representations using recording identifiers. Actor assignments, emotion labels and train/validation/test partitions were verified for consistency.

The following representations were evaluated:

- Transformer-only: 128 dimensions
- Handcrafted + Transformer: 176 dimensions
- Reduced handcrafted + Transformer: 171 dimensions

The downstream classifier was an MLP trained on frozen feature representations.

## Audio Fusion Validation Results

| Representation | Accuracy | Macro-F1 |
|---|---:|---:|
| Transformer-only | 43.75% | 0.3964 |
| Handcrafted + Transformer | **47.92%** | **0.4565** |
| Reduced handcrafted + Transformer | 47.92% | 0.4514 |

The full handcrafted + Transformer representation achieved the highest validation Macro-F1.

## Final Held-Out Test Results

The selected models were evaluated on 240 recordings from four previously unseen actors (02, 03, 08 and 17).

| Representation | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Handcrafted-only | 35.42% | 0.3500 | 0.3512 |
| Transformer-only | 54.58% | 0.5151 | 0.5341 |
| **Handcrafted + Transformer** | **57.50%** | **0.5445** | **0.5683** |

The fused representation improved test Macro-F1 by approximately 0.0293 over Transformer-only and 0.1944 over handcrafted-only.

These results are based on a single actor-independent partition and do not establish statistical significance across different actor splits.

## Model Artifacts

Trained models are stored in `outputs/models/`:

- `handcrafted_only.joblib`
- `transformer_only.joblib`
- `handcrafted_transformer.joblib`

Each artifact contains the fitted scaling/classification pipeline, expected feature-column order and emotion label mapping.

The fused classifier requires both handcrafted features and the corresponding Transformer embeddings. The `.joblib` artifact alone does not extract Transformer features from raw audio.

## Evaluation Outputs

Evaluation files are stored under `outputs/metrics/` and `outputs/figures/`.

These include validation comparisons, final test metrics, per-class classification reports, confusion matrices and sample-level predictions.

## Limitations

- Results were obtained from one fixed actor-independent train/validation/test partition.
- The MLP's internal early-stopping subset was not actor-group-aware.
- The deep-learned audio encoder was provided by Member 2 and treated as frozen.
- No statistical significance testing across multiple actor partitions was performed.
- Final results should be interpreted as experimental findings on the RAVDESS speech subset, not guaranteed real-world emotion-recognition performance.

## 8. Current Implementation Status

| Component | Status |
|---|---|
| Dataset preparation | Completed |
| Actor-independent splitting | Completed |
| Audio preprocessing | Completed |
| Time-domain feature extraction | Completed |
| Spectral feature extraction | Completed |
| Fixed-length feature aggregation | Completed |
| DWT feature extraction | Completed |
| Handcrafted classification | Completed |
| Handcrafted + deep feature fusion | Completed |

## 9. Notes

- Raw RAVDESS audio recordings are not committed to Git.
- Feature scaling for machine-learning classifiers will be fitted using training data only.
- Validation data will be used for model selection.
- Held-out test data will be reserved for final evaluation.
- All project members use the same actor-independent partitions.