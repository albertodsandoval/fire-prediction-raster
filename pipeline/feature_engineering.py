import pandas as pd

RAW_WEATHER_FEATURE_COLUMNS = [
    "tmmx",
    "tmmn",
    "pr",
    "vs",
    "rmax",
    "vpd",
    "erc",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "evt_fuel",
    "evt_class",
]

ENGINEERED_FEATURE_COLUMNS = [
    "temp_t",
    "temp_t_minus_1",
    "temp_t_minus_7",
    "precip_sum_14d",
    "fire_count_90d",
]

NUMERIC_FEATURE_COLUMNS = RAW_WEATHER_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS
MODEL_FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS


def add_lagged_features(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset = dataset.sort_values(["cell_id", "date"]).reset_index(drop=True)

    by_cell = dataset.groupby("cell_id", sort=False)

    dataset["temp_t"] = dataset["tmmx"]
    dataset["temp_t_minus_1"] = by_cell["tmmx"].shift(1)
    dataset["temp_t_minus_7"] = by_cell["tmmx"].shift(7)
    dataset["precip_sum_14d"] = by_cell["pr"].transform(
        lambda values: values.rolling(window=14, min_periods=1).sum()
    )

    if "fire_label_t1" in dataset.columns:
        dataset["fire_count_90d"] = by_cell["fire_label_t1"].transform(
            lambda values: values.shift(1).rolling(window=90, min_periods=1).sum()
        )
        dataset["fire_count_90d"] = dataset["fire_count_90d"].fillna(0).astype(int)

    if "ndvi" in dataset.columns:
        ndvi_rolling_mean = by_cell["ndvi"].transform(
            lambda values: values.rolling(window=30, min_periods=1).mean()
        )
        dataset["ndvi_anomaly_30d"] = dataset["ndvi"] - ndvi_rolling_mean

    return dataset
