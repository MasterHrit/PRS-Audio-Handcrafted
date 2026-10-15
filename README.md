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

## 7. Planned Work

### Day 3 — Wavelet Feature Extraction

- Implement Discrete Wavelet Transform (DWT).
- Extract wavelet subband statistics.
- Combine DWT features with the existing time-domain and spectral representations.

### Day 4 — Feature Analysis and Classification

- Compare handcrafted feature groups.
- Train SVM and MLP classifiers.
- Evaluate using validation Macro-F1 and supporting classification metrics.

### Day 5 — Final Handcrafted Evaluation

- Finalize handcrafted feature representation.
- Evaluate the selected classifier.
- Save model artifacts and experimental results.

### Days 6–7 — Audio Feature Fusion

- Integrate selected deep-learned audio features from Member 2.
- Train a feature-concatenation MLP classifier.
- Compare handcrafted-only, learned-only and combined representations.

## 8. Current Implementation Status

| Component | Status |
|---|---|
| Dataset preparation | Completed |
| Actor-independent splitting | Completed |
| Audio preprocessing | Completed |
| Time-domain feature extraction | Completed |
| Spectral feature extraction | Completed |
| Fixed-length feature aggregation | Completed |
| DWT feature extraction | Planned |
| Handcrafted classification | Planned |
| Handcrafted + deep feature fusion | Planned |

## 9. Notes

- Raw RAVDESS audio recordings are not committed to Git.
- Feature scaling for machine-learning classifiers will be fitted using training data only.
- Validation data will be used for model selection.
- Held-out test data will be reserved for final evaluation.
- All project members use the same actor-independent partitions.