from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from osgeo import gdal
import numpy as np
from xml.etree import ElementTree as ET
from sklearn.tree import export_graphviz
from sklearn import tree
from graphviz import Source
import rasterio
from sklearn.preprocessing import LabelEncoder

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
    for band in range(i.RasterCount):
        data = i.GetRasterBand(band + 1).ReadAsArray(0, 0, nX, nY)
        image_data.append(data)
    image_data = np.array(image_data)
    flattened_image = image_data.flatten()
    images.append(flattened_image)

for label in labels_path:
    tr = ET.parse(label)
    root = tr.getroot()
    for elem in root.iter('name'):
        labels.append(elem.text)

images = np.array(images)
labels = np.array(labels)

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