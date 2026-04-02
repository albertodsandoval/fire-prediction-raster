from rasterio.warp import Resampling

from pipeline.assemble import assemble_dataset
from pipeline.export import (
    export_day_variable_to_geotiff,
    export_grid_cells_to_gpkg,
    export_grid_mask_to_geotiff,
)
from pipeline.firms_source import load_firms_labels
from pipeline.grid import build_cell_lookup, build_target_grid, load_region
from pipeline.gridmet_source import load_gridmet_features
from pipeline.landfire_source import load_landfire_features

CELL_SIZE = 500  # meters
REGION_MODE = "california"
DATES = ("2020-07-01", "2020-07-10")
VARIABLES = ["tmmx", "tmmn", "pr", "vs"]
GRIDMET_SOURCE_MODE = "local_netcdf" if REGION_MODE == "california" else "pygridmet"
GRIDMET_NETCDF_DIR = "D:\\Development\\Datasets\\Sequoia\\gridMET netCDF"
EXPORT_VARIABLE = "tmmx"
EXPORT_DATE = "2020-07-01"
EXPORT_OUTPUT_PATH = f"outputs\\{REGION_MODE}_{EXPORT_VARIABLE}_{EXPORT_DATE}.tif"
GRID_MASK_OUTPUT_PATH = f"outputs\\{REGION_MODE}_grid_mask.tif"
GRID_VECTOR_OUTPUT_PATH = f"outputs\\{REGION_MODE}_grid_cells.gpkg"
RESAMPLING_BY_VARIABLE = {
    "tmmx": Resampling.bilinear,
    "tmmn": Resampling.bilinear,
    "pr": Resampling.bilinear,
    "vs": Resampling.bilinear,
}


def main() -> None:
    region = load_region(REGION_MODE)
    print(region)

    transform, width, height, mask = build_target_grid(region, CELL_SIZE)
    cell_lookup = build_cell_lookup(mask, transform)

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
    labels_df = load_firms_labels(weather_df)
    dataset = assemble_dataset(cell_lookup, weather_df, static_df, labels_df)
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

    dataset.to_csv('output.csv', index=False)   

    print(dataset.head())
    print(f"Exported grid mask to {grid_mask_path}.")
    print(f"Exported grid cells to {grid_vector_path}.")
    print(f"Exported {EXPORT_VARIABLE} for {EXPORT_DATE} to {export_path}.")
    print(
        f"Processed {len(weather_df['date'].unique())} day(s) across {len(cell_lookup)} grid cell(s) "
        f"for {REGION_MODE} with variables {VARIABLES}."
    )


if __name__ == "__main__":
    main()
