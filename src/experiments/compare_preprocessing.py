
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = PROJECT_ROOT / "outputs/features/combined"
OUTPUT_DIR = PROJECT_ROOT / "outputs/metrics"

WAVELETS = ("haar", "db2", "db4", "sym4", "db8")
FRAME_LENGTHS = (20, 25, 30)

METADATA_COLUMNS = [
    "sample_id",
    "actor_id",
    "emotion_label",
    "split",
]


def evaluate_configuration(
    wavelet: str,
    normalize: bool,
    frame_ms: int,
) -> dict:
    """Train SVC and evaluate one preprocessing configuration."""

    # 1. Load the corresponding dataset
    norm_label = "norm" if normalize else "raw"

    file_path = (
        FEATURE_DIR
        / wavelet
        / f"combined_{norm_label}_{frame_ms}ms.csv"
    )

    dataset = pd.read_csv(file_path)

    # 2. Identify the 48 feature columns
    feature_columns = [
        col for col in dataset.columns
        if col not in METADATA_COLUMNS
    ]

    # 3. Separate training and validation recordings
    train_df = dataset[dataset["split"] == "train"]
    val_df = dataset[dataset["split"] == "val"]

    X_train = train_df[feature_columns]
    y_train = train_df["emotion_label"]

    X_val = val_df[feature_columns]
    y_val = val_df["emotion_label"]

    # 4. Create classifier pipeline
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
        ),
    )

    # 5. Train classifier
    model.fit(X_train, y_train)

    # 6. Predict validation emotions
    y_pred = model.predict(X_val)

    # 7. Calculate evaluation metrics
    accuracy = accuracy_score(y_val, y_pred)

    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "wavelet": wavelet,
        "normalization": norm_label,
        "frame_ms": frame_ms,
        "n_features": len(feature_columns),
        "val_accuracy": accuracy,
        "val_macro_f1": macro_f1,
    }


if __name__ == "__main__":

    results = []

    # Evaluate all 30 configurations
    for wavelet in WAVELETS:
        for normalize in (False, True):
            for frame_ms in FRAME_LENGTHS:

                result = evaluate_configuration(
                    wavelet=wavelet,
                    normalize=normalize,
                    frame_ms=frame_ms,
                )

                results.append(result)

                print(
                    f"{result['wavelet']} | "
                    f"{result['normalization']} | "
                    f"{result['frame_ms']}ms | "
                    f"Macro-F1: {result['val_macro_f1']:.4f}"
                )

    # 8. Create results DataFrame
    results_df = pd.DataFrame(results)

    # 9. Rank configurations by Macro-F1
    results_df = results_df.sort_values(
        by="val_macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    # 10. Save comparison results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR / "logreg_preprocessing_comparison.csv"
    )

    results_df.to_csv(output_path, index=False)

    # 11. Display top 10 configurations
    print("\nTop 10 preprocessing configurations:")
    print(results_df.head(10).to_string(index=False))

    # 12. Display best configuration
    best = results_df.iloc[0]

    print("\nBest preprocessing configuration:")
    print("Wavelet:", best["wavelet"])
    print("Normalization:", best["normalization"])
    print("Frame length:", best["frame_ms"], "ms")
    print("Validation Accuracy:", round(best["val_accuracy"], 4))
    print("Validation Macro-F1:", round(best["val_macro_f1"], 4))

    print("\nSaved:", output_path)
