from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from osgeo import gdal
import numpy as np
from xml.etree import ElementTree as ET
from sklearn.tree import export_graphviz
from sklearn import tree
from graphviz import Source

# Load your Pascal VOC dataset and preprocess the data
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

# X should contain the features and y should contain the corresponding labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.5, random_state=60)

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

confusionmatrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:", confusionmatrix)

feature_names=[f"feature_{i}" for i in range(len(X_train[0]))]

class_names=np.unique(y_train)


dot_data = export_graphviz(tree_estimator,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True)

y_test.value_counts()

graph = Source(dot_data)
graph.view()
