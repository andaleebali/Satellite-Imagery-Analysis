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

# Loads Pascal VOC dataset and preprocesses the data
mapfile = '/home/s1885898/scratch/data/training_16_bit/map.txt'
image_path = []
labels_path = []

with open(mapfile) as file:
    for line in file.readlines():
        element = line.strip('\n')
        element = element.replace('\\', '/')
        element = element.split()
        image_path.append('/home/s1885898/scratch/data/training_16_bit/' + element[0])
        labels_path.append('/home/s1885898/scratch/data/training_16_bit/' + element[1])

images = []
labels = []

for image in image_path:
    i = gdal.Open(image)
    nX = i.RasterXSize
    nY = i.RasterYSize
    image_data = []
    step_size = 50
    datas = np.zeros([step_size, step_size, 4])

    rgb=np.zeros([step_size, step_size, 3])



    nir = i.GetRasterBand(4).ReadAsArray() / 32767.0
    red = i.GetRasterBand(3).ReadAsArray() / 32767.0
    green = i.GetRasterBand(2).ReadAsArray() / 32767.0
    blue= i.GetRasterBand(1).ReadAsArray() / 32767.0



    datas[:,:,0] = red 
    datas[:,:,1] = green
    datas[:,:,2] = blue
    datas[:,:,3] = nir


    rgb[:,:,0] = red 
    rgb[:,:,1] = green
    rgb[:,:,2] = blue

    flattened_image = datas.flatten()
    
    images.append(flattened_image)

plt.imshow(rgb)
plt.show()

for label in labels_path:
    tr = ET.parse(label)
    root = tr.getroot()
    for elem in root.iter('name'):
        labels.append(elem.text)

images = np.array(images)
labels = np.array(labels)

# Visualize 5 sample images
num_samples = 5
fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

for i in range(num_samples):
    image_show = images[i].reshape(50, 50, 4)
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=60)

# Encode the labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Create a Random Forest classifier
clf = RandomForestClassifier()

# Train the classifier
clf = clf.fit(X_train, y_train)

tree_estimator = clf.estimators_[0]

# Predict the labels for the test set
y_pred = clf.predict(X_test)

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
    image_show = X_test[i].reshape(50, 50, 4)
    label = y_test[i]
    pred_label = y_pred[i]
    
    # Display the image
    axes[i].imshow(image_show[:, :, :3])  # Display only RGB bands, excluding NIR
    axes[i].set_title(f"True: {label}\nPredicted: {pred_label}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

