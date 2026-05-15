import pandas as pd


def assemble_dataset(
    cell_lookup: pd.DataFrame,
    weather_df: pd.DataFrame,
    static_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    dataset = cell_lookup.merge(weather_df, on="cell_id", how="inner")
    dataset = dataset.merge(static_df, on="cell_id", how="left")
    dataset = dataset.merge(labels_df, on=["cell_id", "date"], how="left")
    dataset["fire_label_t1"] = dataset["fire_label_t1"].fillna(0).astype(int)
    return dataset
