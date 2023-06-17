import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt

# Read the GeoTIFF image
image_path = '/home/s1885898/scratch/data/Subset319.TIF'
with rasterio.open(image_path) as dataset:
    image = dataset.read()
    profile = dataset.profile.copy()

# Read the shapefile
shapefile_path = 'masks_100.shp'
shapes = gpd.read_file(shapefile_path)

# Rasterize the shapefile to create a binary mask
mask = rasterize(shapes.geometry, out_shape=image.shape[1:], transform=dataset.transform)

# Expand the mask to have four bands
expanded_mask = np.repeat(np.expand_dims(mask, axis=0), 4, axis=0)

# Apply the mask to the image for segmentation
segmented_image = np.where(expanded_mask, image, 0)

# Save the segmented image as a GeoTIFF
output_path = 'segmented_image.tif'
profile.update(count=segmented_image.shape[0], compress='lzw')

with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(segmented_image)

# Plot the segmented image
plt.imshow(segmented_image[0], cmap='gray')  # Assuming you want to visualize the first band
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Segmented Image')
plt.show()
