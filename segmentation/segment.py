import rasterio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the new image
new_image_path = '/home/s1885898/scratch/data/202306031515133123358/images/000000000000.tif'
new_image = rasterio.open(new_image_path)
bands = new_image.read()

# Preprocess the new image (assuming it has the same shape as the training data)
preprocessed_image = np.transpose(bands, [1, 2, 0])

# Normalize the preprocessed image (if necessary)
preprocessed_image = preprocessed_image / 255.0  # normalize to [0, 1]

patchsize = 256
stride = 256
patches = []

for r in range(0, preprocessed_image.shape[0] - patchsize + 1, stride):
    for c in range(0, preprocessed_image.shape[1] - patchsize + 1, stride):
        patch = preprocessed_image[r:r+patchsize, c:c+patchsize, :]
        patches.append(patch)

input_shape = (patchsize, patchsize, bands.shape[0])  # Update the input shape based on the number of bands

new_model = tf.keras.models.load_model('model.h5')

new_model.compile(loss='categorical_crossentropy', optimizer='Adam')  # Specify the appropriate loss and optimizer

new_model.summary()

patches_array = np.array(patches)  # Convert the list of patches to a NumPy array

# Reshape the patches_array to match the expected input shape of the model
patches_array = patches_array.reshape((-1,) + input_shape)

predictions = new_model.predict(patches_array)

# Threshold the predictions (if necessary)
threshold = 0.5
thresholded_predictions = (predictions > threshold).astype(np.uint8)

# Plot the original image (RGB bands) if available
if preprocessed_image.shape[2] >= 3:
    plt.subplot(1, 3, 1)
    plt.imshow(preprocessed_image[:, :, :3]) # Display RGB bands
    plt.title('Original Image (RGB)')

# Plot the NIR band if available
if preprocessed_image.shape[2] >= 4:
    plt.subplot(1, 3, 2)
    plt.imshow(preprocessed_image[:, :, 3], cmap='gray')  # Display NIR band
    plt.title('NIR Band')

plt.subplot(1, 3, 3)
plt.imshow(thresholded_predictions[0, :, :, 0], cmap='gray')
plt.title('Predicted Mask')

plt.tight_layout()
plt.show()

