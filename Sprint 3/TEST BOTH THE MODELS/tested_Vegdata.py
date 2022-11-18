
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
import pandas as pd
import cv2
from time import sleep


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")
global img

def KNN():
    global img
    dataset = pd.read_csv("leaf_disease.csv")
    print(dataset)
    x = dataset.iloc[:,:-1] #independent
    y = dataset.iloc[:,-1] #dependent 
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.25, random_state=0)
    print(X_train)
    print(Y_train)
    print(X_test)
    print(Y_test)
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, Y_train)
    Y_predict = classifier.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(Y_test, Y_predict))
    print(classification_report(Y_test, Y_predict))
    from sklearn import metrics
    #Model Acc555uracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(Y_test, Y_predict))
    img = cv2.resize(img,(400,400))
    cv2.imshow("Original Frame",img)
    ## convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv",hsv)
    ## mask of red (36,0,0) ~ (70, 255,255)
    mask1 = cv2.inRange(hsv, (0,0,100), (0,0,255)) #red
    #cv2.imshow("mask1",mask1)
    red= cv2.countNonZero(mask1)
    print("red = ",red)
    img = cv2.GaussianBlur(img,(5,5),2)
    im_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(im_gray,127,255,0)
    count = cv2.countNonZero(thresh)
    #print(count)
    RED=((red+count)/2)*0.001000
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(im_gray, contours, -1, (0,255,0), 6)
        cv2.imshow("contour",im_gray)
    output = classifier.predict([[red]])
    print("Predicted New Output = ",output)
    if output == 1:
        print("Affected")
        print("Total Percentage of Affected = ",int(RED))
    if output == 0:
        print("Normal")
      
def classify(img_file):
    global img
    img_name = img_file
    print(img_name)
    test_image = image.load_img(img_name, target_size = (512,512))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    print(result[0][0])
    if result[0][0] == 0:
        prediction = 'Corn Affected'
        img = cv2.imread(img_name)
        KNN()
    else:
        prediction = 'Corn Normal'
    print(prediction,img_name)
    

import os
path = 'data/test'
files = []
print(path)
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
   for file in f:
     if '.jpg' in file:
       files.append(os.path.join(r, file))
       
for f in files:
   classify(f)
   print('\n')
