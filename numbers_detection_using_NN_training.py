import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizer_v2.adam import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.models import load_model
import pickle
#####################
path = 'myData'
testRatio = 0.2
valRatio = 0.2
imageDimensions = (32,32,3)
#####################
count = 0
Images = []
classNo = []
myList = os.listdir(path)
print("Total number of classes detected is", len(myList))
nOfClasses = len(myList)
print("Importing images...")
for x in range(0,nOfClasses):
    myPicList = os.listdir(path+"/"+str(x))
    #print(myPicList)
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg,(imageDimensions[0],imageDimensions[1]))
        Images.append(curImg)
        classNo.append(count)
    print(count,end=" ")
    count += 1
print(" ")
print("Total images in Images list is", len(Images))
print("Total IDs in classNo list is", len(classNo))

Images = np.array(Images)
classNo = np.array(classNo)
print(Images.shape)
#print(classNo.shape)

#Spliting the data

x_train,x_test,y_train,y_test = train_test_split(Images,classNo,test_size=testRatio)
x_train,x_validation,y_train,y_validation = train_test_split(x_train,y_train,test_size=valRatio)

print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

nofSamples = []
for x in range(0,nOfClasses):
    #print(len(np.where(y_train == x)[0]))
    nofSamples.append(len(np.where(y_train == x)[0]))
print(nofSamples)

plt.figure(figsize=(10,5))
plt.bar(range(0,nOfClasses),nofSamples)
plt.title("number of images for each class")
plt.xlabel("classID")
plt.ylabel("number of images")
plt.show()

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

#img = preProcessing(x_train[30])
#img = cv2.resize(img,(300,300))
#cv2.imshow("Preprocessed",img)
#cv2.waitKey(0)

x_train = np.array(list(map(preProcessing,x_train)))
x_test = np.array(list(map(preProcessing,x_test)))
x_validation = np.array(list(map(preProcessing,x_validation)))


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_validation = x_validation.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(x_train)

y_train = to_categorical(y_train,nOfClasses)
y_test = to_categorical(y_test,nOfClasses)
y_validation = to_categorical(y_validation,nOfClasses)

def myModel():
    nofFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    nOfNodes = 500

    model = Sequential()
    model.add((Conv2D(nofFilters,sizeOfFilter1,input_shape=(imageDimensions[0],
                                                            imageDimensions[1],
                                                            1),activation='relu')))
    model.add((Conv2D(nofFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(nofFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(nofFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(nOfNodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nOfClasses,activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())

Batch_size_val = 50
epochsVal = 10
stepPerEpoch = len(x_train)//Batch_size_val

history = model.fit_generator(dataGen.flow(x_train,y_train,
                    batch_size=Batch_size_val),
                    steps_per_epoch=stepPerEpoch,
                    epochs=epochsVal,
                    validation_data=(x_validation,y_validation),
                    shuffle=1)

plt.figure(1)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(['training','validation'])
plt.title("loss")
plt.xlabel("epoch")

plt.figure(2)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(['training','validation'])
plt.title("Accuracy")
plt.xlabel("epoch")
plt.show()
score = model.evaluate(x_test,y_test,verbose=0)
print("test score", score[0])
print("test accuracy", score[1])

model.save("myModel.h5")
