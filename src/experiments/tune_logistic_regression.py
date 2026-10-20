
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "outputs/features/combined/db2/combined_norm_20ms.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs/metrics"

METADATA_COLUMNS = [
    "sample_id",
    "actor_id",
    "emotion_label",
    "split",
]

RANDOM_STATE = 42


def tune_logistic_regression():

    # 1. Load dataset
    dataset = pd.read_csv(DATASET_PATH)

    feature_columns = [
        col for col in dataset.columns
        if col not in METADATA_COLUMNS
    ]

    # 2. Separate training and validation
    train_df = dataset[dataset["split"] == "train"]
    val_df = dataset[dataset["split"] == "val"]

    X_train = train_df[feature_columns]
    y_train = train_df["emotion_label"]
    groups = train_df["actor_id"]

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

    # 3. Define pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logisticregression", LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ])

    # 4. Hyperparameter grid
    param_grid = {
        "logisticregression__C": [
            0.001,
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
        ],
    }

    # 5. Actor-independent cross-validation
    cv = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # 6. Grid search
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=2,
        verbose=1,
        refit=True,
        return_train_score=True,
    )

    # 7. Train using training actors only
    search.fit(
        X_train,
        y_train,
        groups=groups,
    )

    # 8. Best CV results
    print("\nBest CV Macro-F1:", search.best_score_)
    print("Best hyperparameters:", search.best_params_)

    # 9. Evaluate on validation actors
    best_model = search.best_estimator_

    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)

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

    print("\nTraining Macro-F1:", train_macro_f1)
    print("Validation Accuracy:", val_accuracy)
    print("Validation Macro-F1:", val_macro_f1)

    # 10. Save all hyperparameter results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(search.cv_results_)

    output_path = (
        OUTPUT_DIR / "logreg_grid_search_results.csv"
    )

    results_df.to_csv(output_path, index=False)

    # 11. Display all six configurations
    columns = [
        "rank_test_score",
        "param_logisticregression__C",
        "mean_train_score",
        "mean_test_score",
        "std_test_score",
    ]

    print("\nLogistic Regression hyperparameter comparison:")

    print(
        results_df[columns]
        .sort_values("rank_test_score")
        .to_string(index=False)
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    tune_logistic_regression()
