#Vggnet model
# import the packages/libaries
from keras.models import Sequential
# from keras.layers.normalization import BatchNormalization
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.initializers import TruncatedNormal
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class SimpleVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1


        model.add(Conv2D(32, (3, 3), padding="same",
            input_shape=inputShape,kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        # (CONV => RELU) * 3 => POOL
        model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same",kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25))

        # FC layer
        model.add(Flatten())
        model.add(Dense(512,kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.6))

        # softmax
        model.add(Dense(classes,kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.01)))
        model.add(Activation("softmax"))

        return model

# train model
# import necessary package
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2

# read data and labels
print("------start reading ------")
data = []
labels = []
import pandas as pd
# get the path of image data
imageNames = pd.read_csv('label.csv')
#
for imageName,imageLabel in zip(imageNames['file_name'],imageNames['label']):
    print(imageName)
    # read image data
    image = cv2.imread('./image/'+imageName)
    image = cv2.resize(image, (64, 64))
    data.append(image)
    # read label
    labels.append(imageLabel)

# scale the image
import pandas as pd
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# data split
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# converts label to one-hot encoding format
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# strength the data
aug = ImageDataGenerator( width_shift_range=0.1,
                         height_shift_range=0.1)

# build conv network（64*64*3）
model = SimpleVGGNet.build(width=64, height=64, depth=3, classes=len(lb.classes_))
model.summary()
#
INIT_LR = 0.01
EPOCHS = 100
BS = 32

#
print("------prepare training model------")
# opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer='SGD', metrics=["accuracy"])

# train model
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                        validation_steps=0.2, steps_per_epoch=len(trainX) // BS,
                        epochs=EPOCHS)
"""
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    epochs=EPOCHS, batch_size=32)
"""

# test
print("------test------")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1)))
# lb.classes_
# plot training loss and accuracy curve
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch ")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
# plt.savefig('./output/cnn_plot.png')

# save model
print("------saving model------")
model.save('cnn.model')
f = open('cnn_lb.pickle', "wb")
f.write(pickle.dumps(lb))
f.close()


#load model to make prediction
# import the packages/libraries
from keras.models import load_model
import pickle
import cv2

# load test data
image = cv2.imread('IMAGE_0002.jpg')#read data
output = image.copy()
image = cv2.resize(image, (64, 64))#set the size of picture

# scale the image data
image = image.astype("float") / 255.0

# reshape the image
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# load model and label
print("------load model and label------")
model = load_model('cnn.model')
lb = pickle.loads(open('cnn_lb.pickle', "rb").read(),encoding='iso-8859-1')

# Make prediction
preds = model.predict(image)

# get prediction results
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

# draw results in the test picture
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# drawing
cv2.imshow("Image", output)
cv2.waitKey(0)