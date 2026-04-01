import pandas as pd


def load_landfire_features(cell_lookup: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for static LANDFIRE rasters projected onto the shared grid."""
    return pd.DataFrame({"cell_id": cell_lookup["cell_id"].values})
