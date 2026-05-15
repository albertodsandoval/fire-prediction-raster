from pathlib import Path
from datetime import datetime
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline.feature_engineering import (
    CATEGORICAL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    add_lagged_features,
)

DATA_PATH = Path("output_fuel_names.csv")
TARGET_COLUMN = "fire_label_t1"
FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS
TEST_FRACTION = 0.2
RANDOM_STATE = 42
# "random_forest", "logistic_regression", or "xgboost"
MODEL_TYPE = "xgboost"
BALANCE_TRAINING = False
XGBOOST_SCALE_POS_WEIGHT = 1.0
TRAINING_DISTRIBUTION_CHART_PATH = Path(
    "outputs/model_training_class_distribution.svg")


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset: {path}")

    dataset = pd.read_csv(path)
    required_columns = ["date", TARGET_COLUMN, "tmmx", "pr"]
    missing_columns = [
        column for column in required_columns if column not in dataset.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing {missing_columns}. "
            "Regenerate output.csv with the current preprocessing code."
        )

    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset = add_lagged_features(dataset)

    missing_features = [
        column for column in FEATURE_COLUMNS if column not in dataset.columns
    ]
    if missing_features:
        raise ValueError(f"""Dataset is missing model features {
                         missing_features}.""")

    return dataset


def temporal_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(dataset["date"].dt.date.unique()))
    if len(dates) < 2:
        raise ValueError("Need at least two dates for a train/test split.")

    split_index = max(1, int(len(dates) * (1 - TEST_FRACTION)))
    split_index = min(split_index, len(dates) - 1)
    train_dates = set(dates[:split_index])

    train_df = dataset[dataset["date"].dt.date.isin(train_dates)].copy()
    test_df = dataset[~dataset["date"].dt.date.isin(train_dates)].copy()
    return train_df, test_df


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURE_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURE_COLUMNS),
        ],
        verbose_feature_names_out=False,
    )


def build_model() -> Pipeline:
    if MODEL_TYPE == "logistic_regression":
        return Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                (
                    "logistic_regression",
                    LogisticRegression(max_iter=1000),
                ),
            ]
        )

    if MODEL_TYPE == "random_forest":
        return Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                (
                    "random_forest",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=5,
                        class_weight=None if BALANCE_TRAINING else "balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    if MODEL_TYPE == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "MODEL_TYPE='xgboost' requires the xgboost package. "
                "Install project dependencies with `pip install -r requirements.txt`."
            ) from exc

        return Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                (
                    "xgboost",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="aucpr",
                        tree_method="hist",
                        scale_pos_weight=XGBOOST_SCALE_POS_WEIGHT,
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    raise ValueError(
        "MODEL_TYPE must be 'random_forest', 'logistic_regression', or 'xgboost'."
    )


def balance_training_rows(train_df: pd.DataFrame) -> pd.DataFrame:
    positives = train_df[train_df[TARGET_COLUMN] == 1]
    negatives = train_df[train_df[TARGET_COLUMN] == 0]

    if positives.empty or negatives.empty:
        raise ValueError(
            "Training split needs both positive and negative labels.")

    sampled_negatives = negatives.sample(
        n=len(positives),
        random_state=RANDOM_STATE,
    )
    return (
        pd.concat([positives, sampled_negatives], ignore_index=True)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def print_class_balance(name: str, y: pd.Series) -> None:
    positives = int(y.sum())
    total = len(y)
    print(f"{name}: {positives:,} positives / {total:,} rows ({positives / total:.3%})")


def _pie_wedge_path(cx: int, cy: int, radius: int, start_angle: float, end_angle: float) -> str:
    x1 = cx + radius * math.cos(start_angle)
    y1 = cy + radius * math.sin(start_angle)
    x2 = cx + radius * math.cos(end_angle)
    y2 = cy + radius * math.sin(end_angle)
    large_arc = 1 if end_angle - start_angle > math.pi else 0
    return (
        f"M {cx} {cy} L {x1:.3f} {y1:.3f} "
        f"A {radius} {radius} 0 {large_arc} 1 {x2:.3f} {y2:.3f} Z"
    )


def write_training_distribution_pie(y_train: pd.Series) -> Path | None:
    counts = {
        "Non-fire": int((y_train == 0).sum()),
        "Fire": int((y_train == 1).sum()),
    }
    total = sum(counts.values())
    colors = {"Non-fire": "#4F6F8F", "Fire": "#D95F02"}
    cx, cy, radius = 260, 230, 125
    start_angle = -math.pi / 2

    slices = []
    for label, count in counts.items():
        if count == 0:
            continue
        if count == total:
            slices.append(
                f'<circle cx="{cx}" cy="{cy}" r="{
                    radius}" fill="{colors[label]}"/>'
            )
            continue
        end_angle = start_angle + (count / total) * math.tau
        slices.append(
            f'<path d="{_pie_wedge_path(
                cx, cy, radius, start_angle, end_angle)}" '
            f'fill="{colors[label]}" stroke="#ffffff" stroke-width="2"/>'
        )
        start_angle = end_angle

    legend_rows = []
    legend_y = 185
    for label, count in counts.items():
        pct = 100 * count / total if total else 0
        legend_rows.append(
            f'<rect x="470" y="{legend_y - 13}" width="14" height="14" '
            f'fill="{colors[label]}"/>'
        )
        legend_rows.append(
            f'<text x="495" y="{legend_y}" font-size="18" fill="#202124">'
            f'{label}: {count:,} ({pct:.3f}%)</text>'
        )
        legend_y += 34

    model_name = MODEL_TYPE.replace("_", " ").title()
    balance_mode = "balanced" if BALANCE_TRAINING else "unbalanced"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="460" viewBox="0 0 760 460">
  <rect width="760" height="460" fill="#ffffff"/>
  <text x="380" y="42" text-anchor="middle" font-size="24" font-weight="700" fill="#202124">{model_name} Training Class Distribution</text>
  <text x="380" y="74" text-anchor="middle" font-size="15" fill="#5f6368">BALANCE_TRAINING={BALANCE_TRAINING} ({balance_mode}); target={TARGET_COLUMN}; rows={total:,}</text>
  {''.join(slices)}
  {''.join(legend_rows)}
</svg>
'''

    TRAINING_DISTRIBUTION_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        TRAINING_DISTRIBUTION_CHART_PATH.write_text(svg, encoding="utf-8")
        return TRAINING_DISTRIBUTION_CHART_PATH
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = TRAINING_DISTRIBUTION_CHART_PATH.with_name(
            f"{TRAINING_DISTRIBUTION_CHART_PATH.stem}_{timestamp}"
            f"{TRAINING_DISTRIBUTION_CHART_PATH.suffix}"
        )
        try:
            fallback_path.write_text(svg, encoding="utf-8")
            return fallback_path
        except PermissionError:
            return None


def print_threshold_metrics(y_true: pd.Series, y_probability: np.ndarray) -> None:
    y_predicted = (y_probability >= 0.5).astype(int)

    print("\nConfusion matrix at threshold 0.5:")
    print(confusion_matrix(y_true, y_predicted))
    print("\nClassification report at threshold 0.5:")
    print(classification_report(y_true, y_predicted, zero_division=0))

    if y_true.nunique() == 2:
        print(f"ROC AUC: {roc_auc_score(y_true, y_probability):.4f}")
        print(f"PR AUC:  {average_precision_score(y_true, y_probability):.4f}")
    else:
        print("ROC AUC and PR AUC skipped because the test set has one class.")


def print_top_risk_recall(y_true: pd.Series, y_probability: np.ndarray) -> None:
    ranked = pd.DataFrame({"actual": y_true.to_numpy(),
                          "probability": y_probability})
    ranked = ranked.sort_values("probability", ascending=False)
    positives = ranked["actual"].sum()

    print("\nRecall if reviewing highest-risk cells:")
    for fraction in (0.01, 0.05):
        n_rows = max(1, int(len(ranked) * fraction))
        found = ranked.head(n_rows)["actual"].sum()
        recall = found / positives if positives else 0
        print(
            f"Top {fraction:.0%}: found "
            f"{int(found):,}/{int(positives):,} positives ({recall:.3%})"
        )


def _transformed_feature_names(model: Pipeline) -> np.ndarray:
    return model.named_steps["preprocessor"].get_feature_names_out()


def print_model_importance(model: Pipeline, max_rows: int = 30) -> None:
    feature_names = _transformed_feature_names(model)

    if MODEL_TYPE == "logistic_regression":
        coefficients = model.named_steps["logistic_regression"].coef_[0]
        coefficient_table = pd.DataFrame(
            {"feature": feature_names, "coefficient": coefficients}
        )
        coefficient_table["abs_coefficient"] = coefficient_table["coefficient"].abs(
        )
        coefficient_table = coefficient_table.sort_values(
            "abs_coefficient", ascending=False
        )

        print(
            f"\nTop {max_rows} logistic regression coefficients by absolute value:")
        print(coefficient_table[["feature", "coefficient"]].head(max_rows).to_string(
            index=False))
        return

    estimator_name = "xgboost" if MODEL_TYPE == "xgboost" else "random_forest"
    importances = model.named_steps[estimator_name].feature_importances_
    importance_table = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    model_name = MODEL_TYPE.replace("_", " ").title()
    print(f"\nTop {max_rows} {model_name} feature importances:")
    print(importance_table.head(max_rows).to_string(index=False))


def main() -> None:
    dataset = load_dataset(DATA_PATH)
    train_df, test_df = temporal_split(dataset)
    training_df = balance_training_rows(
        train_df) if BALANCE_TRAINING else train_df

    x_train = training_df[FEATURE_COLUMNS]
    y_train = training_df[TARGET_COLUMN].astype(int)
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].astype(int)

    print_class_balance("Original train", train_df[TARGET_COLUMN].astype(int))
    print_class_balance("Model train", y_train)
    print_class_balance("Test", y_test)
    print(f"Model: {MODEL_TYPE}")
    print(f"Balance training: {BALANCE_TRAINING}")
    chart_path = write_training_distribution_pie(y_train)
    if chart_path is None:
        print(
            "Training class distribution chart skipped: chart output path is not writable.")
    else:
        print(f"Training class distribution chart: {chart_path}")

    model = build_model()
    model.fit(x_train, y_train)

    y_probability = model.predict_proba(x_test)[:, 1]
    print_threshold_metrics(y_test, y_probability)
    print_top_risk_recall(y_test, y_probability)
    print_model_importance(model)


if __name__ == "__main__":
    main()
