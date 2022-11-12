import numpy as np
import cv2
from keras.models import load_model
import pickle

################################
width = 640
height = 480
tresh_hold = 0.65
################################
cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

#pickle_in = open("model_trained.p","rb")
#model = pickle.load(pickle_in)
model_new = load_model("myModel.h5")
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

while True:
    success, img_original = cap.read()
    img = np.asarray(img_original)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    cv2.imshow("Processes Image",img)
    img = img.reshape(1,32,32,1)
    #Prediction
    ClassIndex = int(model_new.predict_classes(img))
    #print(ClassIndex)
    Predictions = model_new.predict(img)
    #print(Predictions)
    probVal = np.amax(Predictions)
    print(ClassIndex,probVal)

    if probVal > tresh_hold:
        cv2.putText(img_original,str(ClassIndex) + "  " + str(probVal),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)

    cv2.imshow("Original Image",img_original)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break



