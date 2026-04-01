import pandas as pd


def load_firms_labels(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for daily FIRMS point labels rasterized onto the shared grid."""
    labels = weather_df[["cell_id", "date"]].drop_duplicates().copy()
    labels["fire_label"] = 0
    return labels
