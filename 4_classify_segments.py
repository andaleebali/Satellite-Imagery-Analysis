import geopandas as gpd
import numpy as np
from shapely.geometry import shape
import rasterio
from rasterio.features import geometry_mask
import joblib
from matplotlib import pyplot as plt
from rasterio.mask import mask
from rasterio.plot import show

import time
start = time.time()


shppath = '/home/s1885898/scratch/data/buildings.shp'
shapefile = gpd.read_file(filename=shppath)

print(shapefile.crs)

tiffile = '/home/s1885898/scratch/data/Subset3_16_bit.tif'
opentif = rasterio.open(fp=tiffile)

shapefile = shapefile.to_crs(opentif.crs)

model_path = 'rf.sav'
loaded_model = joblib.load(model_path)

# Create a new attribute column for predictions
shapefile['predictions'] = None
count=0

for index, row in shapefile.iterrows():
    geometry = row.geometry
    print(geometry)
    geometry_list = [geometry]  # Create a list containing the single geometry
    masked_img, out_transfirm = mask(shapes=geometry_list, dataset=opentif, crop=True)
    clipped_tiff = opentif.read(masked=True)[0]
    #show(masked_img,3)
    #plt.show()
    normalise_tiff = masked_img/ 32767.0
    resized_tiff = np.resize(masked_img, (1, 10000))  # Resize to 10,000 features
    #plt.imshow(resized_tiff)
    #plt.show()
    
    predictions = loaded_model.predict(resized_tiff)
    prediction = predictions[0,0]
    shapefile.at[index, 'predictions'] = prediction  # Assign the prediction to the new attribute
    print(count, prediction)
    count +=1

# Save the updated shapefile
output_shapefile_path = '/home/s1885898/scratch/data/OneDrive_1_15-06-2023/buildings_predictions.shp'
shapefile.to_file(output_shapefile_path)
opentif.close()

end = time.time()
totaltime = end - start
print ("\n" + str(totaltime))