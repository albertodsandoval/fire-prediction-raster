from rasterio.warp import Resampling

from pipeline.assemble import assemble_dataset
from pipeline.export import (
    export_day_variable_to_geotiff,
    export_grid_cells_to_gpkg,
    export_grid_mask_to_geotiff,
)
from pipeline.feature_engineering import add_lagged_features
from pipeline.firms_source import load_firms_labels
from pipeline.grid import build_cell_lookup, build_target_grid, grid_bounds, load_region
from pipeline.gridmet_source import load_gridmet_features
from pipeline.landfire_source import load_landfire_features

CELL_SIZE = 5000  # meters
REGION_MODE = "california"
DATES = ("2020-07-01", "2020-07-31")
VARIABLES = ["tmmx", "tmmn", "pr", "vs", "rmax", "vpd", "erc"]
GRIDMET_SOURCE_MODE = "local_netcdf" if REGION_MODE == "california" else "pygridmet"
GRIDMET_NETCDF_DIR = "D:\\Development\\Datasets\\Sequoia\\gridMET netCDF"
EXPORT_VARIABLE = "tmmx"
EXPORT_DATE = "2020-07-01"
EXPORT_OUTPUT_PATH = f"""outputs\\{REGION_MODE}_{
    EXPORT_VARIABLE}_{EXPORT_DATE}.tif"""
GRID_MASK_OUTPUT_PATH = f"outputs\\{REGION_MODE}_grid_mask.tif"
GRID_VECTOR_OUTPUT_PATH = f"outputs\\{REGION_MODE}_grid_cells.gpkg"
RESAMPLING_BY_VARIABLE = {
    "tmmx": Resampling.bilinear,
    "tmmn": Resampling.bilinear,
    "pr": Resampling.bilinear,
    "vs": Resampling.bilinear,
    "rmax": Resampling.bilinear,
    "vpd": Resampling.bilinear,
    "erc": Resampling.bilinear,
}


def main() -> None:
    region = load_region(REGION_MODE)
    print(region)

    transform, width, height, mask = build_target_grid(region, CELL_SIZE)
    cell_lookup = build_cell_lookup(mask, transform)
    print("GRID BOUNDS:", grid_bounds(transform, width, height))
    weather_df = load_gridmet_features(
        region=region,
        dates=DATES,
        variables=VARIABLES,
        transform=transform,
        width=width,
        height=height,
        mask=mask,
        cell_lookup=cell_lookup,
        resampling_by_variable=RESAMPLING_BY_VARIABLE,
        source_mode=GRIDMET_SOURCE_MODE,
        netcdf_dir=GRIDMET_NETCDF_DIR,
    )
    static_df = load_landfire_features(cell_lookup)
    labels_df = load_firms_labels(
        firms_path="assests\\fire_archive_SV-C2_750007.shp",
        cell_lookup=cell_lookup,
        transform=transform,
        width=width,
        height=height,
        dates=DATES,
    )
    dataset = assemble_dataset(cell_lookup, weather_df, static_df, labels_df)
    dataset = add_lagged_features(dataset)
    grid_mask_path = export_grid_mask_to_geotiff(
        mask=mask,
        transform=transform,
        output_path=GRID_MASK_OUTPUT_PATH,
    )
    grid_vector_path = export_grid_cells_to_gpkg(
        cell_lookup=cell_lookup,
        transform=transform,
        cell_size=CELL_SIZE,
        output_path=GRID_VECTOR_OUTPUT_PATH,
    )
    export_path = export_day_variable_to_geotiff(
        dataset=dataset,
        variable=EXPORT_VARIABLE,
        date=EXPORT_DATE,
        transform=transform,
        width=width,
        height=height,
        output_path=EXPORT_OUTPUT_PATH,
    )

    fire_mask = dataset["fire_label_t1"] == 1
    non_fire_mask = dataset["fire_label_t1"] == 0

    print("\n=== LABEL DEBUG BEFORE CSV SAVE ===")
    print("labels_df sum:", labels_df["fire_label_t1"].sum())
    print("dataset sum:", dataset["fire_label_t1"].sum())
    print("total rows:", len(dataset))
    print("non-fire label rows:", int(non_fire_mask.sum()))
    print("fire label rows:", int(fire_mask.sum()))
    print("unique non-fire cells:",
          dataset.loc[non_fire_mask, "cell_id"].nunique())
    print("unique fire cells:", dataset.loc[fire_mask, "cell_id"].nunique())

    print("\nValue counts:")
    print(dataset["fire_label_t1"].value_counts().sort_index())

    print("\nSample positives:")
    print(dataset[fire_mask].head())

    print("\nUnique cells with fire:")
    print(dataset.loc[fire_mask, "cell_id"].nunique())
    print(dataset.head())
    print("\nAttempting to save output.csv...")
    dataset.to_csv('output.csv', index=False)
    print("Saved output.csv.")
    print(f"Exported grid mask to {grid_mask_path}.")
    print(f"Exported grid cells to {grid_vector_path}.")
    print(f"Exported {EXPORT_VARIABLE} for {EXPORT_DATE} to {export_path}.")
    print(
        f"""Processed {len(weather_df['date'].unique())} day(s) across {
            len(cell_lookup)} grid cell(s) """
        f"""for {REGION_MODE} with variables {VARIABLES}."""
    )


if __name__ == "__main__":
    main()
