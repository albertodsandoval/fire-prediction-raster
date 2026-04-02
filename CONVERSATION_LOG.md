# Conversation Log

Purpose: maintain a concise record of meaningful user-agent exchanges for this repository.

Logging rules:
- Record only exchanges that led to code changes, file changes, persistent repo documentation changes, or a specific implementation constraint that materially affected later work.
- Skip routine questions, one-off diagnostics, and conversational turns that did not change the repo or the way work should be done.
- Keep entries short and decision-oriented, not transcript-style.
- When an exchange changes behavior for future sessions, note both the request and the resulting policy.

## 2026-03-31

### Persistent session memory in-repo
- The user asked for an `.md` file that preserves meaningful back-and-forth across sessions.
- The user specified that trivial exchanges should be omitted unless they explain why something was handled a certain way.
- Result: added this log file and an `AGENTS.md` instruction file so future agents maintain the log in the requested sparse format.

### Switched feature extraction from polygon zonal stats to a raster-grid workflow
- The user asked how to make statewide processing scale better than the original Sequoia-oriented polygon workflow and then requested implementation of the mask, destination array, and reprojection changes.
- Decision: replace per-day GeoDataFrame cell generation plus `zonal_stats` over temporary TIFFs with a fixed 500 m target raster grid, a one-time rasterized study-area mask, and in-memory reprojection of each day onto that grid.
- Result: `main.py` now supports explicit `california` vs `sequoia` region selection, filters the state shapefile correctly before GridMET requests, and extracts values from masked reprojected arrays rather than polygon summaries.

### Expanded the raster-grid workflow to multiple GridMET variables
- The user asked to extend the new Sequoia-first raster workflow beyond a single `tmmx` variable and requested an explanation of the changes.
- Decision: keep one shared target grid, mask, and cell lookup table, but loop through a configurable list of GridMET variables and reproject each variable-day slice onto the same grid before adding it as a feature column.
- Result: `main.py` now requests multiple GridMET variables in one dataset, applies per-variable reprojection on the shared grid, and outputs one row per cell-day with multiple weather feature columns.

### Scaffolded a modular shared-grid pipeline
- The user asked for a first-pass project structure for multi-source pipelining because the shared-grid workflow was still concentrated in `main.py`.
- Decision: split the pipeline into modules for grid setup, GridMET weather loading, dataset assembly, and placeholder adapters for LANDFIRE and FIRMS so new sources can conform to the same `cell_id` and `date` keys.
- Result: added a `pipeline/` package, moved the reusable grid and GridMET logic into modules, reduced `main.py` to orchestration, and established placeholders for static raster features and daily fire labels.

### Added tiled GridMET download support for California-scale runs
- The user wanted to process all of California even though single statewide GridMET requests were failing.
- Decision: split the requested region into intersecting geographic tiles, call `pygridmet.get_bygeom()` per tile, and mosaic each tile back onto the shared statewide raster grid during reprojection.
- Result: `pipeline/grid.py` now builds request tiles, `pipeline/gridmet_source.py` supports tiled downloads with actionable tile-failure errors, and `main.py` can enable California tiling by setting `GRIDMET_TILE_SIZE_DEG`.

### Refined GridMET tiling to avoid clipped coastal sliver requests
- The first California tile still failed even at 1-degree tiling because the request geometry was clipped to a very small coastal sliver before `pygridmet` converted it to a bounding box.
- Decision: use padded rectangular request boxes for any tile that intersects California, rather than passing clipped sliver geometries into `get_bygeom()`.
- Result: the GridMET tiling path now sends simpler geographic boxes with a configurable padding margin, which should produce more stable NCSS requests along the coast.

### Switched California GridMET ingestion to local yearly NetCDF files
- After repeated failures from valid, tiled `pygridmet.get_bygeom()` requests for California, the user decided to move to local NetCDF ingestion for statewide weather features.
- Decision: keep `pygridmet` for smaller park-scale jobs, but for California load yearly GridMET NetCDF files from disk, subset them locally by date and California bounds, and then reproject onto the shared statewide analysis grid.
- Result: `pipeline/gridmet_source.py` now supports `source_mode='local_netcdf'`, and `main.py` defaults California runs to local NetCDF files such as `tmmx_2020.nc`, `tmmn_2020.nc`, `pr_2020.nc`, and `vs_2020.nc`.

### Added GeoTIFF export for presentation-ready raster visualization
- The user wanted to visualize generated cell-day data as a raster for presentation rather than only keeping it in tabular form.
- Decision: reconstruct a raster from the assembled dataset using the stored `row` and `col` indices on the shared grid, then write a single-band GeoTIFF for a chosen variable and date.
- Result: added `pipeline/export.py` and wired `main.py` to export an example GeoTIFF such as `outputs\\california_tmmx_2020-07-01.tif` after each run.

### Added one-time exports for the shared analysis grid
- The user wanted the analysis grid itself exported once, both as a raster and as vector cells, for presentation and inspection.
- Decision: export the boolean grid mask as a GeoTIFF and reconstruct valid cell polygons from `row`/`col` indices into a GeoPackage layer.
- Result: `main.py` now writes files such as `outputs\\california_grid_mask.tif` and `outputs\\california_grid_cells.gpkg` alongside the weather raster export.

### Added a shapefile-loading workaround for missing `.shx` sidecars
- A California run failed because `CA_State.shp` was present but its `.shx` index file was missing or unreadable, causing `pyogrio` to refuse the dataset.
- Decision: route shared shapefile reads through a helper that sets `SHAPE_RESTORE_SHX=YES` so GDAL can rebuild the missing index when possible.
- Result: `pipeline/grid.py` now uses `_read_shapefile(...)` for both state and park boundaries, which should make region loading more resilient to incomplete shapefile sidecars.

### Added a preprocessing script to create California-only GridMET NetCDFs
- The user wanted to clip the downloaded GridMET yearly NetCDF files to California once so later statewide runs would avoid repeatedly loading full-CONUS inputs.
- Decision: add a standalone script that subsets yearly GridMET NetCDFs by California bounds, optionally clips them to the California polygon, and writes reusable `*_california.nc` outputs.
- Result: added `scripts/clip_gridmet_to_california.py`, which can produce files such as `tmmx_2020_california.nc`, `tmmn_2020_california.nc`, `pr_2020_california.nc`, and `vs_2020_california.nc`.

### Made the GridMET preprocessing script runnable as a direct script
- Running `python scripts\\clip_gridmet_to_california.py` failed because the `pipeline` package was not on `sys.path` when executed as a standalone script.
- Decision: have the script prepend the repo root to `sys.path` before importing from `pipeline`.
- Result: `scripts\\clip_gridmet_to_california.py` should now run directly from the repository root without requiring `python -m` invocation.

### Fixed CRS propagation in the California NetCDF clipping script
- The preprocessing script failed at `rio.clip(...)` because the selected DataArray no longer carried CRS metadata after subsetting.
- Decision: explicitly restore the spatial dims and write `EPSG:4326` onto the subset before clipping it to the California polygon.
- Result: `scripts\\clip_gridmet_to_california.py` now calls `rio.set_spatial_dims(...).rio.write_crs(4326)` on the subset prior to the shape clip.
