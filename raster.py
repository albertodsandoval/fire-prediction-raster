import geopandas as gpd
import numpy as np
from shapely.geometry import box
import rasterio
from rasterio.features import rasterize

# park boundary (already projected, e.g. EPSG:3310)
park = gpd.read_file("D:/Development/Datasets/Sequoia/nps_boundary/nps_boundary.shp")
park = park.to_crs("EPSG:3310")

# bounding box
minx, miny, maxx, maxy = park.total_bounds

cell_size = 500  # meters

# generate grid
cells = []
for x in np.arange(minx, maxx, cell_size):
    for y in np.arange(miny, maxy, cell_size):
        cells.append(box(x, y, x + cell_size, y + cell_size))

grid = gpd.GeoDataFrame(geometry=cells, crs=park.crs)

# clip to park
grid = grid[grid.intersects(park.union_all())].copy()

# assign ids + centroids
grid["cell_id"] = range(len(grid))
grid["cx"] = grid.centroid.x
grid["cy"] = grid.centroid.y
