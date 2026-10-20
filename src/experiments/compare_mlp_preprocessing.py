
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning


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
    """Train MLP and evaluate one preprocessing configuration."""

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

    # Basic checks
    assert len(feature_columns) == 48
    assert len(train_df) == 960
    assert len(val_df) == 240

    assert set(train_df["actor_id"]).isdisjoint(
        set(val_df["actor_id"])
    )

    assert np.isfinite(X_train.to_numpy()).all()
    assert np.isfinite(X_val.to_numpy()).all()

    # 4. Create MLP pipeline
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128,64),
            activation="relu",
            solver="adam",
            alpha=1,
            learning_rate_init=0.0001,
            max_iter=400,
            early_stopping=False,
            random_state=42,
        ),
    )

    # 5. Train classifier
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train, y_train)

    converged = not any(
        issubclass(w.category, ConvergenceWarning)
        for w in caught
    )

    # 6. Predict training and validation emotions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    # 7. Calculate evaluation metrics
    train_macro_f1 = f1_score(
        y_train,
        train_pred,
        average="macro",
        zero_division=0,
    )

    val_accuracy = accuracy_score(y_val, val_pred)

    val_macro_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0,
    )

    val_weighted_f1 = f1_score(
        y_val,
        val_pred,
        average="weighted",
        zero_division=0,
    )

    # Access trained MLP
    mlp = model.named_steps["mlpclassifier"]

    return {
        "wavelet": wavelet,
        "normalization": norm_label,
        "frame_ms": frame_ms,
        "n_features": len(feature_columns),
        "train_macro_f1": train_macro_f1,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "val_weighted_f1": val_weighted_f1,
        "n_iter": mlp.n_iter_,
        "final_loss": mlp.loss_,
        "converged": converged,
    }


if __name__ == "__main__":

    results = []

    # Evaluate all 30 configurations
    # for wavelet in WAVELETS:
    #     for normalize in (False, True):
    #         for frame_ms in FRAME_LENGTHS:
    for wavelet in ("db2",):
        for normalize in (False,):
            for frame_ms in (20,):

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
                    f"Train F1: {result['train_macro_f1']:.4f} | "
                    f"Val F1: {result['val_macro_f1']:.4f} | "
                    f"Iterations: {result['n_iter']} | "
                    f"Converged: {result['converged']}"
                )

    # 8. Create results DataFrame
    results_df = pd.DataFrame(results)

    # 9. Rank configurations by validation Macro-F1
    results_df = results_df.sort_values(
        by="val_macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    # 10. Save comparison results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR / "mlp_preprocessing_comparison.csv"
    )

    results_df.to_csv(output_path, index=False)

    # 11. Display top 10 configurations
    print("\nTop 10 MLP preprocessing configurations:")
    print(results_df.head(10).to_string(index=False))

    # 12. Display best configuration
    best = results_df.iloc[0]

    print("\nBest MLP preprocessing configuration:")
    print("Wavelet:", best["wavelet"])
    print("Normalization:", best["normalization"])
    print("Frame length:", best["frame_ms"], "ms")
    print("Validation Accuracy:", round(best["val_accuracy"], 4))
    print("Validation Macro-F1:", round(best["val_macro_f1"], 4))
    print("Training Macro-F1:", round(best["train_macro_f1"], 4))

    print("\nSaved:", output_path)
