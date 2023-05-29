import os
from samgeo import SamGeo, show_image, download_file, overlay_images, tms_to_geotiff
from rasterio.plot import show
import matplotlib.pyplot as plt
from osgeo import gdal
from PIL import Image
from pathlib import Path
import numpy
import torchvision.transforms as tt

# Open the GeoTIFF file
dataset = gdal.Open('/home/s1885898/scratch/data/3_band_split/Subset3_8_bit_3band0.TIF').ReadAsArray()

# Visualize the raster data on a map
#plt.figure(figsize=(10,10))
#show(dataset)
#plt.show()

#out_dir = os.path.join(os.path.expanduser("~"), "Downloads")
#checkpoint = os.path.join(out_dir, "sam_vit_h_4b8939.pth")

checkpoint = '/home/s1885898/scratch/sam_vit_h_4b8939.pth'

sam = SamGeo(
    model_type="vit_h",
    checkpoint=checkpoint,
    sam_kwargs=None,
    )

data_transpose=numpy.transpose(dataset)
# data_transpose = data_transpose.astype('uint8')

#h = data_transpose.shape[1]
#w = data_transpose.shape[2]
#long_side = h if h > w else w
#scale_factor = 1024.0 / long_side

sam.generate(data_transpose, output="masks.tif", foreground=True, unique=True)
sam.show_masks(cmap="binary_r")