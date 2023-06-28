from sklearn.cluster import KMeans
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.mask import mask
from sklearn.preprocessing import StandardScaler
import pickle
#dataset
tiffile = '/home/s1885898/scratch/data/Subset3.tif'
opentif = rasterio.open(fp=tiffile)

shppath = '/home/s1885898/scratch/data/OneDrive_1_15-06-2023/building_footprints.shp'
shapefile = gpd.read_file(filename=shppath)

print(shapefile.crs)

shapefile = shapefile.to_crs(opentif.crs)

# Create a new attribute column for predictions
shapefile['predictions'] = None
shapefile['skipped'] = False
count=0
resized_tiffs = []
geometries = []

for index, row in shapefile.iterrows():
    try:
        geometry = row.geometry
        if not geometry.is_valid:
            shapefile.at[index, 'skipped'] = True
            continue  # Skip invalid geometries
        
        print(geometry)
        geometry_list = [geometry]  # Create a list containing the single geometry
        masked_img, out_transform = mask(shapes=geometry_list, dataset=opentif, crop=True)
        clipped_tiff = opentif.read(masked=True)[0]
        #show(masked_img, 3)
        #plt.show()
        normalise_tiff = masked_img
        transposed_img = np.transpose(normalise_tiff, (1, 2, 0))
        resized_tiff = np.resize(transposed_img, (1, 10000))  # Resize to 10,000 features
        #plt.imshow(resized_tiff)
        #plt.show()

        resized_tiffs.append(resized_tiff)
        geometries.append(geometry)

        print(count)
        count += 1
    except Exception as e:
        shapefile.at[index, 'skipped'] = True
        print(f"Error processing geometry at index {index}: {e}")
        continue  # Skip to the next iteration if an error occurs

# Convert the lists to arrays
resized_tiffs = np.vstack(resized_tiffs)
resized_tiffs = resized_tiffs.reshape(resized_tiffs.shape[0], -1)  # Reshape to (num_samples, num_features)
geometries = np.array(geometries)

pickle.dump(resized_tiffs, open('tiffs.pkl',"wb"))
pickle.dump(geometries, open('geometries.pkl',"wb"))


#model
model = KMeans(n_clusters=12)

model.fit(resized_tiffs)

# Check the length of the labels array
print(len(model.labels_))
print(len(shapefile))

shapefile.loc[~shapefile['skipped'], 'predictions'] = model.labels_

# Save the updated shapefile
output_shapefile_path = '/home/s1885898/scratch/data/OneDrive_1_15-06-2023/kmeans_predictions.shp'
shapefile.to_file(output_shapefile_path)
opentif.close()

import geopandas as gpd
import matplotlib.pyplot as plt

# Assuming shapefile contains the geometries and predictions column
# Replace 'predictions' with the appropriate column name from your data
cluster_column = 'predictions'

# Plotting the shapefile
shapefile.plot(column=cluster_column, categorical=True, legend=True, figsize=(10, 10), cmap='Set1')
plt.title('Clustering Results')
plt.show()
