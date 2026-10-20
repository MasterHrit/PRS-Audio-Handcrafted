
from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.metrics import accuracy_score, f1_score


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


# 1. Load dataset
dataset = pd.read_csv(DATASET_PATH)

feature_columns = [
    col for col in dataset.columns
    if col not in METADATA_COLUMNS
]

train_df = dataset[dataset["split"] == "train"]
val_df = dataset[dataset["split"] == "val"]

X_train = train_df[feature_columns]
y_train = train_df["emotion_label"]
groups = train_df["actor_id"]

X_val = val_df[feature_columns]
y_val = val_df["emotion_label"]


# 2. Define MLP pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        activation="relu",
        solver="adam",
        max_iter=400,
        early_stopping=False,
        random_state=42,
    )),
])


# 3. Define hyperparameter search space
param_distributions = {
    "mlp__hidden_layer_sizes": [
        (64, 32),
        (128, 64),
    ],
    "mlp__alpha": [
        0.001,
        1.0,
    ],
    "mlp__learning_rate_init": [
        0.0001,
        0.001,
    ],
}


# 4. Actor-independent cross-validation
cv = StratifiedGroupKFold(
    n_splits=4,
    shuffle=True,
    random_state=42,
)


# 5. Randomized hyperparameter search
search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42,
    refit=True,
    return_train_score=True,
)


# 6. Fit using training actors only
search.fit(
    X_train,
    y_train,
    groups=groups,
)


# 7. Display cross-validation results
print("\nBest CV Macro-F1:", search.best_score_)
print("\nBest hyperparameters:")
print(search.best_params_)


# 8. Evaluate selected model on validation actors
best_model = search.best_estimator_

val_pred = best_model.predict(X_val)

print("\nValidation Accuracy:",
      accuracy_score(y_val, val_pred))

print("Validation Macro-F1:",
      f1_score(
          y_val,
          val_pred,
          average="macro",
          zero_division=0,
      ))


# 9. Save search results
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(search.cv_results_)

results_df.to_csv(
    OUTPUT_DIR / "mlp_random_search_results.csv",
    index=False,
)

print("\nSearch results saved!")


# 10. Display top 10 hyperparameter combinations

columns = [
    "rank_test_score",
    "mean_train_score",
    "mean_test_score",
    "std_test_score",
    "param_mlp__hidden_layer_sizes",
    "param_mlp__alpha",
    "param_mlp__learning_rate_init",
]

print("\nTop 10 MLP hyperparameter configurations:")

print(
    results_df[columns]
    .sort_values("rank_test_score")
    .head(10)
    .to_string(index=False)
)
