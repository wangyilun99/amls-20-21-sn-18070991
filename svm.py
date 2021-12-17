# import necessary packages
from sklearn.svm import SVC # SVM model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import random
import pickle
import cv2
import os
from PIL import Image
# read data and labels
print("------Start Reading data------")
data = []
labels = []
import pandas as pd
# get the path of images
# imagePaths = sorted(list(utils_paths.list_images('./原图数据集PNG')))
# random.seed(42)
# random.shuffle(imagePaths)
imageNames = pd.read_csv('label.csv')
# read data and labels
for imageName,imageLabel in zip(imageNames['file_name'],imageNames['label']):
    print(imageName)
    # read image data
    image = cv2.imread('./image/'+imageName)
    image = cv2.resize(image, (64, 64))
    image = image.ravel()
    data.append(image)
    # read labels
    labels.append(imageLabel)

# scale the image data
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split data
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2,random_state=10)

# convert label to one-hot encoding format
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# SVM
clf = SVC(kernel='rbf',C=1)
clf.fit(trainX,trainY.argmax(axis=1))
y_pre = clf.predict(testX)
# test accuracy
print('test accuracy:',accuracy_score(testY.argmax(axis=1), y_pre))
#classification report
from sklearn.metrics import classification_report
print(classification_report(testY.argmax(axis=1), y_pre))
#confusion matrix
from scikitplot.metrics import plot_confusion_matrix
plot_confusion_matrix(testY.argmax(axis=1), y_pre)
#cross validation
from sklearn.model_selection import cross_val_score
score = cross_val_score(clf,testX,testY.argmax(axis=1), cv=5)
print('cv-5 score：',score)
print('cv-5 mean-score：',score.mean())