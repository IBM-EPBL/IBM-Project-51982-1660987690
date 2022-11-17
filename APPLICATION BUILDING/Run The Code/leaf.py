# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'eye.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
import pandas as pd
import cv2
from time import sleep

class Ui_CLASSIFY(object):
    def setupUi(self, CLASSIFY):
        CLASSIFY.setObjectName("CLASSIFY")
        CLASSIFY.resize(1969, 944)
        CLASSIFY.setStyleSheet("background-color: rgb(40, 200, 147);")
        self.centralwidget = QtWidgets.QWidget(CLASSIFY)
        self.centralwidget.setObjectName("centralwidget")
        self.TITTLE = QtWidgets.QLabel(self.centralwidget)
        self.TITTLE.setGeometry(QtCore.QRect(190, 0, 1391, 61))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(False)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.TITTLE.setFont(font)
        self.TITTLE.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.TITTLE.setMouseTracking(False)
        self.TITTLE.setAlignment(QtCore.Qt.AlignCenter)
        self.TITTLE.setWordWrap(False)
        self.TITTLE.setObjectName("TITTLE")
        self.IMAGESHOW = QtWidgets.QLabel(self.centralwidget)
        self.IMAGESHOW.setGeometry(QtCore.QRect(630, 100, 551, 351))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.IMAGESHOW.setFont(font)
        self.IMAGESHOW.setFrameShape(QtWidgets.QFrame.Box)
        self.IMAGESHOW.setFrameShadow(QtWidgets.QFrame.Plain)
        self.IMAGESHOW.setLineWidth(2)
        self.IMAGESHOW.setMidLineWidth(0)
        self.IMAGESHOW.setText("")
        self.IMAGESHOW.setAlignment(QtCore.Qt.AlignCenter)
        self.IMAGESHOW.setObjectName("IMAGESHOW")
        self.BROWSEIMAGE = QtWidgets.QPushButton(self.centralwidget)
        self.BROWSEIMAGE.setGeometry(QtCore.QRect(630, 480, 241, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.BROWSEIMAGE.setFont(font)
        self.BROWSEIMAGE.setObjectName("BROWSEIMAGE")
        self.BROWSEIMAGE_2 = QtWidgets.QPushButton(self.centralwidget)
        self.BROWSEIMAGE_2.setGeometry(QtCore.QRect(950, 480, 231, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.BROWSEIMAGE_2.setFont(font)
        self.BROWSEIMAGE_2.setObjectName("BROWSEIMAGE_2")
        self.PREDICTION = QtWidgets.QLabel(self.centralwidget)
        self.PREDICTION.setGeometry(QtCore.QRect(630, 580, 241, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.PREDICTION.setFont(font)
        self.PREDICTION.setFrameShape(QtWidgets.QFrame.Box)
        self.PREDICTION.setText("")
        self.PREDICTION.setObjectName("PREDICTION")
        self.PERCENTAGE = QtWidgets.QLabel(self.centralwidget)
        self.PERCENTAGE.setGeometry(QtCore.QRect(950, 580, 231, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.PERCENTAGE.setFont(font)
        self.PERCENTAGE.setFrameShape(QtWidgets.QFrame.Box)
        self.PERCENTAGE.setText("")
        self.PERCENTAGE.setObjectName("PERCENTAGE")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(640, 540, 231, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(950, 540, 201, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.MEDICINE = QtWidgets.QLabel(self.centralwidget)
        self.MEDICINE.setGeometry(QtCore.QRect(750, 680, 311, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.MEDICINE.setFont(font)
        self.MEDICINE.setFrameShape(QtWidgets.QFrame.Box)
        self.MEDICINE.setText("")
        self.MEDICINE.setObjectName("MEDICINE")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(830, 640, 231, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        CLASSIFY.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(CLASSIFY)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1969, 26))
        self.menubar.setObjectName("menubar")
        CLASSIFY.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(CLASSIFY)
        self.statusbar.setObjectName("statusbar")
        CLASSIFY.setStatusBar(self.statusbar)

        self.retranslateUi(CLASSIFY)
        QtCore.QMetaObject.connectSlotsByName(CLASSIFY)
        self.BROWSEIMAGE.clicked.connect(self.loadImage)
        self.BROWSEIMAGE_2.clicked.connect(self.classifyFunction)
    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)") # Ask for file
        if fileName: # If the user gives a file
            print(fileName)
            self.file=fileName
            pixmap = QtGui.QPixmap(fileName) # Setup pixmap with the provided image
            pixmap = pixmap.scaled(self.IMAGESHOW.width(), self.IMAGESHOW.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
            self.IMAGESHOW.setPixmap(pixmap) # Set the pixmap onto the label
            self.IMAGESHOW.setAlignment(QtCore.Qt.AlignCenter) # Align the label to center
    def classifyFunction(self):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("model.h5")
        print("Loaded model from disk")
        path2=self.file
        test_image = image.load_img(path2, target_size = (512,512))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)         
        if result[0][0] == 0:
            prediction = 'AFFECTED'
            label2=prediction
            self.PREDICTION.setText(label2)
            sleep(1)
            img = cv2.imread(path2)
            dataset = pd.read_csv("leaf_disease.csv")
            print(dataset)
            x = dataset.iloc[:,:-1] #independent
            y = dataset.iloc[:,-1] #dependent 
            from sklearn.model_selection import train_test_split
            X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.80, random_state=0)
            print(X_train)
            print(Y_train)
            print(X_test)
            print(Y_test)
            from sklearn.neighbors import KNeighborsClassifier
            classifier = KNeighborsClassifier(n_neighbors=5)
            classifier.fit(X_train, Y_train)
            Y_predict = classifier.predict(X_test)        
            img = cv2.resize(img,(400,400))
            #cv2.imshow("Original Frame",img)
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
                #cv2.imshow("contour",im_gray)
            output = classifier.predict([[red]])
            print("Predicted New Output = ",output)
            if output == 1:
                print("Affected")
            RED=int(RED)
            if RED <11:
                RED= int(RED)
            label_2=str(RED)
            self.PERCENTAGE.setText(label_2)
            print("Total Percentage of Affected = ",label_2)
            if (RED>=0 and RED<=15):
                MEDICINE="POTASSIUM CHLORIDE 25 ml SPRAYING"
                self.MEDICINE.setText(MEDICINE)
            if (RED>=16 and RED<=30):
                MEDICINE="UREA 50 ml SPRAYING"
                self.MEDICINE.setText(MEDICINE)
            if (RED>=31 and RED<=50):
                MEDICINE="NPK 70 ml SPRAYING"
                self.MEDICINE.setText(MEDICINE)
            if RED>=50:
                MEDICINE="NPK 95 ml SPRAYING"
                self.MEDICINE.setText(MEDICINE)
        else:
            prediction = 'NORMAL'
            label2=prediction
            red=0
            label_2=str(red)
            self.PREDICTION.setText(label2)
            self.PERCENTAGE.setText(label_2)

    def retranslateUi(self, CLASSIFY):
        _translate = QtCore.QCoreApplication.translate
        CLASSIFY.setWindowTitle(_translate("CLASSIFY", "MainWindow"))
        self.TITTLE.setText(_translate("CLASSIFY", "ARTIFITIAL INTELLIGENCE BASED LEAF DISEASE DETECTION  USING DEEP LEARNING"))
        self.BROWSEIMAGE.setText(_translate("CLASSIFY", "BROWSE IMAGE"))
        self.BROWSEIMAGE_2.setText(_translate("CLASSIFY", "CLASSIFY"))
        self.label.setText(_translate("CLASSIFY", "PREDICTION"))
        self.label_2.setText(_translate("CLASSIFY", "PERCENTAGE"))
        self.label_3.setText(_translate("CLASSIFY", "MEDICINE"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    CLASSIFY = QtWidgets.QMainWindow()
    ui = Ui_CLASSIFY()
    ui.setupUi(CLASSIFY)
    CLASSIFY.show()
    sys.exit(app.exec_())
