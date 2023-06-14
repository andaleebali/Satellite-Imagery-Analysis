#File to read a whole tiff and classify it

import gdal
import numpy as np
import joblib

# Function to read a tiff into one 3D array instead of in 30x30 chunks (no batches)
def readTiff(file):
  dataset=gdal.Open(file)
  # read data from geotiff object
  numX=dataset.RasterXSize                   # number of pixels in x direction
  numY=dataset.RasterYSize                   # number of pixels in y direction
  print('Total pixels = ', numX,numY)
  # geolocation tiepoint
  transform_ds = dataset.GetGeoTransform() # extract geolocation information
  xOrigin=transform_ds[0]             # coordinate of x corner
  yOrigin=transform_ds[3]             # coordinate of y corner 
  pixelWidth=transform_ds[1]          # resolution in x direction
  pixelHeight=transform_ds[5]         # resolution in y direction

  # Geospatial information for later writing outputs to tiff
  geotransform = (xOrigin, pixelWidth, 0, yOrigin, 0, pixelHeight)  
  
  data = np.zeros([numY,numX,dataset.RasterCount], dtype=float)  

  NIR=dataset.GetRasterBand(4).ReadAsArray()
  blue=dataset.GetRasterBand(3).ReadAsArray()
  green=dataset.GetRasterBand(2).ReadAsArray()
  red=dataset.GetRasterBand(1).ReadAsArray()

  data[:,:,0] = red 
  data[:,:,1] = green
  data[:,:,2] = blue
  data[:,:,3] = NIR

  
  
  return (data, geotransform, numX, numY)

def tiff_batches(image):

    step_size = img_size
    count = 0

    images=[]
    
    i = gdal.Open(image)
    nX = i.RasterXSize
    nY = i.RasterYSize
    
    numberofrows = int(nX/step_size)
    numberofcolumns = int(nY/step_size)

    print('number of rows:', numberofrows, 'number of columns: ', numberofcolumns)

    geolocation = i.GetGeoTransform()

    xOrig = geolocation[0]

    yOrigin = geolocation[3]

    data2 = np.zeros([numberofrows*numberofcolumns, step_size*step_size*4])

    for row in range(numberofrows):
        for column in range(numberofcolumns):
            datas = np.zeros([step_size, step_size,4])
            nir=i.GetRasterBand(4).ReadAsArray(xoff=row*step_size ,yoff=column*step_size ,win_xsize=step_size, win_ysize=step_size)
            blue=i.GetRasterBand(3).ReadAsArray(xoff=row*step_size ,yoff=column*step_size ,win_xsize=step_size, win_ysize=step_size)
            green=i.GetRasterBand(2).ReadAsArray(xoff=row*step_size ,yoff=column*step_size ,win_xsize=step_size, win_ysize=step_size)
            red=i.GetRasterBand(1).ReadAsArray(xoff=row*step_size ,yoff=column*step_size ,win_xsize=step_size, win_ysize=step_size)

            datas[:,:,0] = red 
            datas[:,:,1] = green
            datas[:,:,2] = blue
            datas[:,:,3] = nir

            datas=datas.flatten()
        
            data2[count,:]=datas
            count +=1
    
    return data2

# Call on above function
root='segmented_image.tif'
M, M_geotransform, numX, numY = readTiff(root)
print(M_geotransform)
img_size=50

# Call on function to convert tiff to batched 4D numpy array
data= tiff_batches(root)
print('Shape of data: ', data.shape)
model_path = 'rf.sav'
loaded_model = joblib.load(model_path)
# From saved model, create predictions list on big dataset and convert to labels(max prediction)
predictions = loaded_model.predict(data)
predict_labels = []
for p in predictions:
  predict_labels.append(np.argmax(p))

label_9_count = 0

for p in predict_labels:
    if p == 2:
        label_9_count +=1

print('Predicted red roofs:', label_9_count)


count = 0

label_mask = np.zeros([data.shape[0], img_size, img_size], dtype=float)

for l in predict_labels:
    label_mask[count,:,:].fill(l)

import matplotlib.pyplot as plt

# Assuming label_mask is a 3D NumPy array with shape (num_samples, img_size, img_size)

# Select a sample from the label mask
sample_index = 0
label_sample = label_mask[sample_index]

# Cast label_sample to integer type
label_sample = label_sample.astype(int)

# Define a color map for visualizing the labels
cmap = plt.cm.get_cmap('viridis', np.max(label_sample) + 1)

# Plot the label mask
plt.imshow(label_sample, cmap=cmap)
plt.colorbar()

# Set axis labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Label Mask')

# Show the plot
plt.show()
