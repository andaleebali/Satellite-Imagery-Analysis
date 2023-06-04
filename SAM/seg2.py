
import os
from samgeo import SamGeo, show_image, download_file, overlay_images, tms_to_geotiff
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy
from osgeo import gdal 
import torchvision.transforms as tt
from PIL import Image
from pathlib import Path
path = '/home/s1885898/scratch/data/Subset3.tif'
d = Image.open(path)
print(d.format, d.size, d.mode)


plt.figure(figsize=(10,10))
show(d, cmap='viridis')
plt.show()


out_dir = os.path.join(os.path.expanduser("~"), "Downloads")
checkpoint = os.path.join(out_dir, "sam_vit_h_4b8939.pth")

sam = SamGeo(
    model_type="vit_h",
    checkpoint=checkpoint,
    sam_kwargs=None,
)

sam.generate(img_tensor, output="masks.tif", foreground=True, unique=True)
sam.show_masks(cmap="binary_r")