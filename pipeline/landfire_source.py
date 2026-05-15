from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as transform_coords

from pipeline.grid import TARGET_CRS

LANDFIRE_EVT_RASTER_PATH = Path(
    "assests/LF2022_EVT_CONUS/Tif/LF2022_EVT_CONUS.tif"
)
LANDFIRE_EVT_LOOKUP_PATH = Path(
    "assests/LF2022_EVT_CONUS/CSV_Data/LF2022_EVT.csv"
)


def _read_evt_lookup(lookup_path: Path) -> pd.DataFrame:
    if not lookup_path.exists():
        raise FileNotFoundError(f"Could not find LANDFIRE EVT lookup: {lookup_path}")

    lookup = pd.read_csv(
        lookup_path,
        usecols=["VALUE", "EVT_FUEL_N", "EVT_CLASS"],
    )
    lookup = lookup.rename(
        columns={
            "VALUE": "evt_value",
            "EVT_FUEL_N": "evt_fuel",
            "EVT_CLASS": "evt_class",
        }
    )
    lookup["evt_value"] = pd.to_numeric(lookup["evt_value"], errors="coerce")
    lookup = lookup.dropna(subset=["evt_value"]).copy()
    lookup["evt_value"] = lookup["evt_value"].astype(int)
    return lookup.drop_duplicates(subset=["evt_value"])


def _sample_evt_values(
    raster_path: Path,
    cell_lookup: pd.DataFrame,
) -> pd.Series:
    if not raster_path.exists():
        raise FileNotFoundError(f"Could not find LANDFIRE EVT raster: {raster_path}")

    required_columns = {"cell_id", "cx", "cy"}
    missing_columns = required_columns.difference(cell_lookup.columns)
    if missing_columns:
        raise ValueError(
            f"cell_lookup is missing required columns: {sorted(missing_columns)}"
        )

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"LANDFIRE EVT raster has no CRS: {raster_path}")

        xs, ys = transform_coords(
            TARGET_CRS,
            src.crs,
            cell_lookup["cx"].to_numpy(),
            cell_lookup["cy"].to_numpy(),
        )
        samples = src.sample(zip(xs, ys), indexes=1, masked=True)
        values = []
        for sample in samples:
            if np.ma.is_masked(sample[0]):
                values.append(pd.NA)
            else:
                value = int(sample[0])
                values.append(pd.NA if value == src.nodata else value)

    return pd.Series(values, index=cell_lookup.index, dtype="Int64")


def load_landfire_features(cell_lookup: pd.DataFrame) -> pd.DataFrame:
    """Sample LANDfire Existing Vegetation Type at each shared-grid cell center."""
    lookup = _read_evt_lookup(LANDFIRE_EVT_LOOKUP_PATH)
    evt_values = _sample_evt_values(LANDFIRE_EVT_RASTER_PATH, cell_lookup)

    features = pd.DataFrame(
        {
            "cell_id": cell_lookup["cell_id"].values,
            "evt_value": evt_values.to_numpy(),
        }
    )
    features = features.merge(lookup, on="evt_value", how="left")
    return features[["cell_id", "evt_fuel", "evt_class"]]
