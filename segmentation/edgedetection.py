import cv2
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

def load_geotiff(filename):
    dataset = gdal.Open(filename)
    raster = dataset.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    return raster

def save_geotiff(filename, data, driver):
    rows, cols = data.shape[:2]
    output = driver.Create(filename, cols, rows, 1, gdal.GDT_Byte)
    output.GetRasterBand(1).WriteArray(data)
    output.FlushCache()

image_path = "NDVI_Subset319_1_2.tif"
output_path = "path_to_save_output.tif"

# Load GeoTIFF image
image = load_geotiff(image_path)

# Check number of channels
num_channels = image.shape[2] if len(image.shape) == 3 else 1

if num_channels == 3:  # RGB image
    rgb_image = image[:, :, :3]
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
elif num_channels == 1:  # Grayscale image
    gray_image = image.squeeze()
else:
    raise ValueError("Unsupported number of channels in the image.")

# Apply edge detection
edges = cv2.Canny(gray_image, 210, 255)

# Visualize the original image and the detected edges
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.title("Edges")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save the resulting image as a GeoTIFF file
driver = gdal.GetDriverByName("GTiff")
save_geotiff(output_path, edges, driver)
