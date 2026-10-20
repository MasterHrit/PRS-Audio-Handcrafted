
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_validate,
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
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "outputs/features/combined/db2/combined_raw_20ms.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs/metrics"

METADATA_COLUMNS = [
    "sample_id",
    "actor_id",
    "emotion_label",
    "split",
]

RANDOM_STATE = 42


def get_classifiers():
    """Return baseline classifiers."""

    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        ),

        "SVC": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale"),
        ),

        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5),
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "AdaBoost": AdaBoostClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),

        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1.0,
                learning_rate_init=0.0001,
                max_iter=400,
                early_stopping=False,
                random_state=RANDOM_STATE,
            ),
        ),
    }


def compare_classifiers():

    # 1. Load dataset
    dataset = pd.read_csv(DATASET_PATH)

    feature_columns = [
        col for col in dataset.columns
        if col not in METADATA_COLUMNS
    ]

    assert len(feature_columns) == 48

    # 2. Separate training and validation
    train_df = dataset[dataset["split"] == "train"]
    val_df = dataset[dataset["split"] == "val"]

    assert len(train_df) == 960
    assert len(val_df) == 240

    assert set(train_df["actor_id"]).isdisjoint(
        set(val_df["actor_id"])
    )

    X_train = train_df[feature_columns]
    X_val = val_df[feature_columns]

    # Encode emotion labels for all classifiers
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(
        train_df["emotion_label"]
    )

    y_val = encoder.transform(
        val_df["emotion_label"]
    )

    groups = train_df["actor_id"]

    # 3. Actor-independent cross-validation
    cv = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    classifiers = get_classifiers()

    results = []

    # 4. Train and evaluate each classifier
    for name, model in classifiers.items():

        print(f"\nTraining: {name}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            cv_results = cross_validate(
                model,
                X_train,
                y_train,
                groups=groups,
                cv=cv,
                scoring="f1_macro",
                return_train_score=True,
                n_jobs=1,
            )

        # 5. Train on all training actors
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # 6. Calculate metrics
        result = {
            "classifier": name,
            "cv_macro_f1_mean": np.mean(
                cv_results["test_score"]
            ),
            "cv_macro_f1_std": np.std(
                cv_results["test_score"]
            ),
            "cv_train_f1_mean": np.mean(
                cv_results["train_score"]
            ),
            "train_macro_f1": f1_score(
                y_train,
                train_pred,
                average="macro",
                zero_division=0,
            ),
            "val_accuracy": accuracy_score(
                y_val,
                val_pred,
            ),
            "val_macro_f1": f1_score(
                y_val,
                val_pred,
                average="macro",
                zero_division=0,
            ),
            "val_weighted_f1": f1_score(
                y_val,
                val_pred,
                average="weighted",
                zero_division=0,
            ),
        }

        results.append(result)

        print(
            f"CV F1: {result['cv_macro_f1_mean']:.4f} | "
            f"Val F1: {result['val_macro_f1']:.4f}"
        )

    return pd.DataFrame(results)


if __name__ == "__main__":

    results_df = compare_classifiers()

    results_df = results_df.sort_values(
        "cv_macro_f1_mean",
        ascending=False,
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR / "classifier_comparison.csv"
    )

    results_df.to_csv(output_path, index=False)

    print("\nClassifier Comparison:")
    print(results_df.to_string(index=False))

    print("\nSaved:", output_path)
