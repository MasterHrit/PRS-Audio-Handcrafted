
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "outputs/features/combined"
OUTPUT_DIR = PROJECT_ROOT / "outputs/metrics/tuning"

# --------------------------------------------------
# Experiment configuration
# --------------------------------------------------

WAVELET = "db2"
NORMALIZATION = "raw"
FRAME_MS = 20

# Select models to tune
MODELS_TO_TUNE = [
    "Logistic Regression",
    "SVC",
    "KNN",
    "Decision Tree",
    "Random Forest",
    "AdaBoost",
    "XGBoost",
    "MLP",
]

RANDOM_STATE = 42
N_SPLITS = 4
N_ITER = 20
N_JOBS = 2

METADATA_COLUMNS = [
    "sample_id",
    "actor_id",
    "emotion_label",
    "split",
]


def get_search_spaces():
    """Define classifiers and hyperparameter search spaces."""

    return {
        "Logistic Regression": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                )),
            ]),
            "params": {
                "model__C": [
                    0.001, 0.01, 0.1, 1.0, 10.0, 100.0
                ],
            },
            "search": "grid",
        },

        "SVC": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC()),
            ]),
            "params": [
                {
                    "model__kernel": ["linear"],
                    "model__C": [0.01, 0.1, 1.0, 10.0],
                },
                {
                    "model__kernel": ["rbf"],
                    "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "model__gamma": [
                        "scale", 0.001, 0.01, 0.1
                    ],
                },
            ],
            "search": "random",
        },

        "KNN": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier()),
            ]),
            "params": {
                "model__n_neighbors": [3, 5, 7, 11, 15],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            "search": "random",
        },

        "Decision Tree": {
            "estimator": DecisionTreeClassifier(
                random_state=RANDOM_STATE,
            ),
            "params": {
                "max_depth": [3, 5, 10, 15, None],
                "min_samples_leaf": [1, 2, 5, 10],
                "criterion": ["gini", "entropy"],
            },
            "search": "random",
        },

        "Random Forest": {
            "estimator": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": ["sqrt", "log2"],
            },
            "search": "random",
        },

        "AdaBoost": {
            "estimator": AdaBoostClassifier(
                random_state=RANDOM_STATE,
            ),
            "params": {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.5, 1.0],
            },
            "search": "grid",
        },

        "XGBoost": {
            "estimator": XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
                tree_method="hist",
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [2, 3, 4, 6],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "reg_lambda": [1.0, 5.0, 10.0],
            },
            "search": "random",
        },

        "MLP": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(
                    activation="relu",
                    solver="adam",
                    max_iter=400,
                    early_stopping=False,
                    random_state=RANDOM_STATE,
                )),
            ]),
            "params": {
                "model__hidden_layer_sizes": [
                    (32,), (64,), (64, 32), (128, 64)
                ],
                "model__alpha": [0.001, 0.01, 0.1, 1.0],
                "model__learning_rate_init": [
                    0.0001, 0.001, 0.01
                ],
            },
            "search": "random",
        },
    }


def load_dataset():

    dataset_path = (
        FEATURE_DIR
        / WAVELET
        / f"combined_{NORMALIZATION}_{FRAME_MS}ms.csv"
    )

    dataset = pd.read_csv(dataset_path)

    feature_columns = [
        col for col in dataset.columns
        if col not in METADATA_COLUMNS
    ]

    train_df = dataset[dataset["split"] == "train"]
    val_df = dataset[dataset["split"] == "val"]
    test_df = dataset[dataset["split"] == "test"]

    assert len(feature_columns) == 48
    assert len(train_df) == 960
    assert len(val_df) == 240
    assert len(test_df) == 240

    assert dataset["sample_id"].is_unique

    assert set(train_df["actor_id"]).isdisjoint(
        set(val_df["actor_id"])
    )
    assert set(train_df["actor_id"]).isdisjoint(
        set(test_df["actor_id"])
    )
    assert set(val_df["actor_id"]).isdisjoint(
        set(test_df["actor_id"])
    )

    X_train = train_df[feature_columns]
    X_val = val_df[feature_columns]

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df["emotion_label"])
    y_val = encoder.transform(val_df["emotion_label"])

    groups = train_df["actor_id"]

    assert np.isfinite(X_train.to_numpy()).all()
    assert np.isfinite(X_val.to_numpy()).all()

    return X_train, y_train, groups, X_val, y_val


def tune_classifier(name, config, X_train, y_train, groups, X_val, y_val):
    """Tune one classifier and evaluate its best model."""

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    common_args = {
        "estimator": config["estimator"],
        "scoring": "f1_macro",
        "cv": cv,
        "n_jobs": N_JOBS,
        "verbose": 1,
        "refit": True,
        "return_train_score": True,
        "error_score": "raise",
    }

    if config["search"] == "grid":
        search = GridSearchCV(
            param_grid=config["params"],
            **common_args,
        )
    else:
        search = RandomizedSearchCV(
            param_distributions=config["params"],
            n_iter=N_ITER,
            random_state=RANDOM_STATE,
            **common_args,
        )

    print(f"\nTuning: {name}")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=ConvergenceWarning,
        )
        search.fit(X_train, y_train, groups=groups)

    best_model = search.best_estimator_

    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)

    best_index = search.best_index_

    result = {
        "classifier": name,
        "wavelet": WAVELET,
        "normalization": NORMALIZATION,
        "frame_ms": FRAME_MS,
        "best_cv_macro_f1": search.best_score_,
        "cv_macro_f1_std": search.cv_results_[
            "std_test_score"
        ][best_index],
        "cv_train_macro_f1": search.cv_results_[
            "mean_train_score"
        ][best_index],
        "train_macro_f1": f1_score(
            y_train, train_pred,
            average="macro", zero_division=0,
        ),
        "val_accuracy": accuracy_score(y_val, val_pred),
        "val_macro_f1": f1_score(
            y_val, val_pred,
            average="macro", zero_division=0,
        ),
        "val_weighted_f1": f1_score(
            y_val, val_pred,
            average="weighted", zero_division=0,
        ),
        "best_params": str(search.best_params_),
    }

    # Save full tuning history for this classifier
    history_path = (
        OUTPUT_DIR
        / f"{name.lower().replace(' ', '_')}_search.csv"
    )

    pd.DataFrame(search.cv_results_).to_csv(
        history_path,
        index=False,
    )

    return result


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, groups, X_val, y_val = load_dataset()

    search_spaces = get_search_spaces()

    results = []

    for name in MODELS_TO_TUNE:

        result = tune_classifier(
            name,
            search_spaces[name],
            X_train,
            y_train,
            groups,
            X_val,
            y_val,
        )

        results.append(result)

        print(
            f"{name} | "
            f"CV F1: {result['best_cv_macro_f1']:.4f} | "
            f"Val F1: {result['val_macro_f1']:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "best_cv_macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    output_path = OUTPUT_DIR / "tuned_classifier_comparison.csv"

    results_df.to_csv(output_path, index=False)

    print("\nTuned Classifier Comparison:")
    print(results_df.to_string(index=False))

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
