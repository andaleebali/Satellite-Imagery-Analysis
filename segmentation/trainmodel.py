import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
import rasterio
import numpy as np
from rasterio.plot import show

def readTiff(mapfile):
    '''
    Reads geotiff file 
    '''

    images = []
    labels = []
    
    with open(mapfile) as file:
        for line in file.readlines():
            element = line.strip('\n')
            element = element.replace('\\','/')
            element = element.split()
            if len(element) == 2:
                images.append('/home/s1885898/scratch/data/202306031515133123358/' + element[0])
                labels.append('/home/s1885898/scratch/data/202306031515133123358/' + element[1])
            else:
                next

    train_images = []
    train_mask = []
    for i in range(0,len(labels)) :
        file = rasterio.open(labels[i])
        bands = file.read()
        train_mask.append(bands)

        file = rasterio.open(images[i])
        bands = file.read()
        train_images.append(bands)
    
    return train_mask, train_images

# Define the U-Net architecture
def unet(input_shape):
    inputs = Input(input_shape)

    # Contracting path
    conv1 = Conv2D(64, 4, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 4, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 4, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 4, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 4, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 4, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 4, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 4, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = Conv2D(1024, 4, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 4, activation='relu', padding='same')(conv5)

    # Expanding path
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(512, 2, activation='relu', padding='same')(up6)
    merge6 = tf.keras.layers.concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 4, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 4, activation='relu', padding='same')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(256, 4, activation='relu', padding='same')(up7)
    merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 4, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 4, activation='relu', padding='same')(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same')(up8)
    merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 4, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 4, activation='relu', padding='same')(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same')(up9)
    merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 4, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 4, activation='relu', padding='same')(conv9)

    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model

train_mask, train_images = readTiff('/home/s1885898/scratch/data/202306031515133123358/map.txt')

#y_train = np.array(y_train)
#X_train = np.array(y_train)

# Define the input shape and create the model
input_shape = (256, 256, 4)
model = unet(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#print(X_train.shape)
#print(y_train.shape)

train_images = np.transpose(train_images, [0,2,3,1])
train_mask = np.transpose(train_mask, [0,2,3,1])

print(train_images.shape)
print(train_mask.shape)

# Train the model (replace X_train and y_train with your training data)
model.fit(train_images, train_mask, batch_size=16, epochs=10, validation_split=0.2)

# Make predictions using the trained model
predictions = model.predict(train_images)

model.save('model.h5')

# Visualize the results
import matplotlib.pyplot as plt

# Select a random sample from the predictions
index = np.random.randint(0, len(train_images))
image = train_images[index]
mask = train_mask[index]
prediction = predictions[index]

# Plot the original image
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original Image')

# Plot the ground truth mask
plt.subplot(1, 3, 2)
plt.imshow(mask[:, :, 0], cmap='gray')
plt.title('Ground Truth Mask')

# Plot the predicted mask
plt.subplot(1, 3, 3)
plt.imshow(prediction[:, :, 0], cmap='gray')
plt.title('Predicted Mask')

plt.tight_layout()
plt.show()