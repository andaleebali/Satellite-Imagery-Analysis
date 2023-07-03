from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from osgeo import gdal
import numpy as np
from xml.etree import ElementTree as ET
from sklearn.tree import export_graphviz
from graphviz import Source
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
from skimage.transform import resize
import cv2
import rasterio

def get_training_data(path_to_data='training_data_rgb'):
    """
    Open the txt file for the training data and extracts the file path for each image and xml file
    """
    mapfile = path_to_data + '/map.txt'
    image_path = []
    label_path = []

    with open(mapfile) as file:
        for line in file.readlines():
            element = line.strip('\n')
            element = element.replace('\\', '/')
            element = element.split()
            image_path.append(path_to_data + '/' + element[0])
            label_path.append(path_to_data + '/' + element[1])

    return image_path, label_path

def preprocess_training_data(image_path, label_path):
    images = []
    labels = []
    for image in image_path:
        nX, nY, red, green, blue, alpha = preprocess_image(image)

        rgb = np.zeros([nX, nY, 3])
        rgb[:, :, 0] = red
        rgb[:, :, 1] = green
        rgb[:, :, 2] = blue

        #plt.imshow(rgb)
        #plt.show()
        

        red_masked = red[alpha != 0]
        green_masked = green[alpha != 0]
        blue_masked = blue[alpha != 0]
    
        red_masked = resize(red_masked, (nX, nY))
        green_masked = resize(green_masked, (nX, nY))
        blue_masked = resize(blue_masked, (nX, nY))

        rgb = np.zeros([nX, nY, 3])
        rgb[:, :, 0] = red_masked
        rgb[:, :, 1] = green_masked
        rgb[:, :, 2] = blue_masked

        #plt.imshow(rgb)
        #plt.show()

        flattened_image = rgb.flatten()
        images.append(flattened_image)

    for label in label_path:
        classification = extract_classification(label)
        labels.append(classification)

    return images, labels


def preprocess_image(image):
    with rasterio.open(image) as dataset:
        nX = dataset.width
        nY = dataset.height

        red = dataset.read(1)
        green = dataset.read(2)
        blue = dataset.read(3)
        alpha = dataset.read(4)

    return nX, nY, red, green, blue, alpha

def extract_classification(label):
    tr = ET.parse(label)
    root = tr.getroot()
    for elem in root.iter('name'):
        classification = elem.text
    return classification


def augment_flip(image_path, label_path):
    images = []
    labels = []
    for image in image_path:
        nX, nY, red, green, blue, alpha = preprocess_image(image)
        
        flipped_red = np.fliplr(red)
        flipped_green = np.fliplr(green)
        flipped_blue = np.fliplr(blue)
        flipped_alpha = np.fliplr(alpha)

        red_masked = flipped_red[flipped_alpha != 0]
        green_masked = flipped_green[flipped_alpha != 0]
        blue_masked = flipped_blue[flipped_alpha != 0]
    
        red_masked = resize(red_masked, (nX, nY))
        green_masked = resize(green_masked, (nX, nY))
        blue_masked = resize(blue_masked, (nX, nY))

        
        rgb = np.zeros([nX, nY, 3])
        
        rgb[:, :, 0] = red_masked
        rgb[:, :, 1] = green_masked
        rgb[:, :, 2] = blue_masked

        print (red[0])
        print(flipped_red[0])

        flattened_image = rgb.flatten()
        images.append(flattened_image)


    for label in label_path:
        classification = extract_classification(label)
        labels.append(classification)

    
    return images, labels


import skimage.io as io
from skimage import exposure

def augment_flip_rotate_equalize(image_path, label_path):
    images = []
    labels = []
    for image in image_path:
        nX, nY, red, green, blue, alpha = preprocess_image(image)
        
        # Flip image horizontally
        flipped_red = np.fliplr(red)
        flipped_green = np.fliplr(green)
        flipped_blue = np.fliplr(blue)
        flipped_alpha = np.fliplr(alpha)

        red_masked = flipped_red[flipped_alpha != 0]
        green_masked = flipped_green[flipped_alpha != 0]
        blue_masked = flipped_blue[flipped_alpha != 0]
    
        red_masked = resize(red_masked, (nX, nY))
        green_masked = resize(green_masked, (nX, nY))
        blue_masked = resize(blue_masked, (nX, nY))

        rgb_flip = np.zeros([nX, nY, 3])
        
        rgb_flip[:, :, 0] = red_masked
        rgb_flip[:, :, 1] = green_masked
        rgb_flip[:, :, 2] = blue_masked

        # Rotate image by 45 degrees
        rotation_angle = 45
        rotation_matrix = cv2.getRotationMatrix2D((nX/2, nY/2), rotation_angle, 1.0)
        rotated_rgb = cv2.warpAffine(rgb_flip, rotation_matrix, (nY, nX))

        flattened_image = rotated_rgb.flatten()
        images.append(flattened_image)


    for label in label_path:
        classification = extract_classification(label)
        labels.append(classification)

    
    return images, labels

image_path, label_path = get_training_data()
images, labels, = preprocess_training_data(image_path, label_path)
augmented_images, augmented_labels = augment_flip(image_path, label_path)
augmented_images2, augmented_labels2 = augment_flip_rotate_equalize(image_path, label_path)

combined_images = []
combined_labels = []

# Add original images and labels
combined_images.extend(images)
combined_labels.extend(labels)

# Add augmented images and labels
combined_images.extend(augmented_images)
combined_labels.extend(augmented_labels)

# Add augmented images and labels
combined_images.extend(augmented_images2)
combined_labels.extend(augmented_labels2)

images = np.array(combined_images)
labels = np.array(combined_labels)

# Visualize 5 sample images
num_samples = 5
fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

for i in range(num_samples):
    image_show = images[i].reshape(50, 50, 3)
    label = labels[i]
    
    # Convert image from BGR to RGB
    image_show_rgb = image_show[..., [0, 1, 2]]
    
    # Normalize pixel values to [0, 1] for 16-bit images
    image_show_rgb = image_show_rgb.astype(float)
    
    # Clip pixel values to [0, 1]
    image_show_rgb = np.clip(image_show_rgb, 0, 1)
    
    # Display the image
    axes[i].imshow(image_show_rgb)  # Display RGB image
    axes[i].set_title(f"Label: {label}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

print(images.shape)
print(labels.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=60)

# Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Create a Random Forest classifier
model = RandomForestClassifier()

# Train the classifier
model = model.fit(X_train, y_train)

tree_estimator = model.estimators_[0]

# Predict the labels for the test set
y_pred = model.predict(X_test)

pickle.dump(model, open('model1.pkl',"wb"))

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

class_names = label_encoder.classes_

confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:", confusion_matrix)

for i in range(len(class_names)):
    print(f"Label: {class_names[i]}")
    for j in range(len(class_names)):
        print(f"Predicted: {class_names[j]}, Count:{confusion_matrix[i, j]}")

feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
class_names = np.unique(y_train)

dot_data = export_graphviz(tree_estimator,
                           feature_names=feature_names,
                           class_names=class_names,
                           filled=True,
                           rounded=True)

graph = Source(dot_data)
graph.render("decision_tree")

# Visualize the test images
num_images = 5  # Number of images to visualize
fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

for i in range(num_images):
    image_show = X_test[i].reshape(50, 50, 3)
    label = y_test[i]
    pred_label = y_pred[i]
    
    # Display the image
    axes[i].imshow(image_show[:, :, :3])  # Display only RGB bands, excluding NIR
    axes[i].set_title(f"True: {label}\nPredicted: {pred_label}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()