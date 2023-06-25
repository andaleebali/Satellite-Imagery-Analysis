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
model = KMeans(n_clusters=5, random_state=90)

# Fit the model to the reshaped data
model.fit(reshaped_data)

# Get the cluster labels for each pixel
labels = model.labels_

# Reshape the labels array back to the original raster shape
labels_reshaped = labels.reshape(transposed_img.shape[:-1])

# Assign unique colors to each cluster label
colors = np.array([
    [0, 32, 46], 
    #[0, 63, 92], 
    [44, 72, 117], 
    #[138, 80, 143], 
    [188,80,144], 
    #[255,99,97],
    [255,133,49],
    #[255,166,0], 
    [255,211,128]])

# Create the classified image
classified_image = colors[labels_reshaped]

# Display the classified image
plt.imshow(classified_image)
plt.axis('off')
plt.show()

grayscale_image = np.dot(classified_image, [0.2989, 0.587, 0.114])
# Get the metadata from the original raster dataset
meta = opentif.meta

# Update the metadata to match the classified image
meta.update(count=1, dtype=str(grayscale_image.dtype))

# Save the classified image as a GeoTIFF
output_file = 'classified_image.tif'
with rasterio.open(output_file, 'w', **meta) as dst:
    dst.write(grayscale_image, 1)

