from sklearn.cluster import KMeans
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Load the raster dataset
tiffile = 'NDVI_Subset319_1_2.tif'
opentif = rasterio.open(tiffile)

# Read the raster data as a numpy array
raster_data = opentif.read()
transposed_img = np.transpose(raster_data, (1, 2, 0))

# Reshape the array to have a single dimension (flatten it)
reshaped_data = transposed_img.reshape(-1, transposed_img.shape[-1])

# Create the K-means clustering model
model = KMeans(n_clusters=4, random_state=10)

# Fit the model to the reshaped data
model.fit(reshaped_data)

# Get the cluster labels for each pixel
labels = model.labels_

# Reshape the labels array back to the original raster shape
labels_reshaped = labels.reshape(transposed_img.shape[:-1])

# Assign unique grayscale values to each cluster label
grayscale_values = np.array([
    0,  # Class 0
    85,  # Class 1
    170,  # Class 2
    200,  # Class 3
    255  # Class 4
])

# Create the grayscale image based on the assigned grayscale values
grayscale_image = grayscale_values[labels_reshaped]

# Get the metadata from the original raster dataset
meta = opentif.meta

# Update the metadata to match the grayscale image
meta.update(count=1, dtype=str(grayscale_image.dtype))

# Save the grayscale image as a GeoTIFF
output_file = 'grayscale_image_test.tif'
with rasterio.open(output_file, 'w', **meta) as dst:
    dst.write(grayscale_image, 1)

# Display the grayscale image
plt.imshow(grayscale_image, cmap='gray')
plt.axis('off')
plt.show()
