from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import mahotas
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd

from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer


main = tkinter.Tk()
main.title("Computer Vision and Machine Learning for Viticulture Technology") #designing main screen
main.geometry("1300x1200")

global filename
global X1, Y1
global X2, Y2
global X3, Y3
global scaler1,scaler2,scaler3
global svm1, svm2, svm3
global X_train1, X_test1, y_train1, y_test1
global X_train2, X_test2, y_train2, y_test2
global X_train3, X_test3, y_train3, y_test3
bins = 8

svm_graph = []
knn_graph = []
lr_graph = []
elm_graph = []

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def uploadDataset(): #function to upload tweeter profile
    text.delete('1.0', END)
    global scaler1,scaler2,scaler3
    global filename
    global X1, Y1
    global X2, Y2
    global X3, Y3    
    filename = filedialog.askdirectory(initialdir = ".")
    text.delete('1.0', END)
    ds1 = np.load('model/dataset1.txt.npy')
    ds1_label = np.load('model/dataset1_label.txt.npy')
    ds2 = np.load('model/dataset2.txt.npy')
    ds2_label = np.load('model/dataset2_label.txt.npy')
    ds3 = np.load('model/dataset3.txt.npy')
    ds3_label = np.load('model/dataset3_label.txt.npy')
    X1 = []
    Y1 = []
    for i in range(len(ds1)):
        image = ds1[i]
        hu = fd_hu_moments(image)
        haralick   = fd_haralick(image)
        histogram  = fd_histogram(image)
        global_feature = np.hstack([histogram, haralick, hu])
        Y1.append(ds1_label[i])
        X1.append(global_feature)
    Y1 = np.asarray(Y1)
    X1 = np.asarray(X1)
    print(X1.shape)
    print(Y1.shape)
    targetNames = np.unique(Y1)
    le          = LabelEncoder()
    Y1      = le.fit_transform(Y1)
    scaler1            = MinMaxScaler(feature_range=(0, 1))
    X1 = scaler1.fit_transform(X1)
    print(Y1)
    print(X1.shape)
    print(Y1.shape)
    X2 = []
    Y2 = []
    for i in range(len(ds2)):
        image = ds2[i]
        hu = fd_hu_moments(image)
        haralick   = fd_haralick(image)
        histogram  = fd_histogram(image)
        global_feature = np.hstack([histogram, haralick, hu])
        Y2.append(ds2_label[i])
        X2.append(global_feature)
    Y2 = np.asarray(Y2)
    X2 = np.asarray(X2)
    print(X2.shape)
    print(Y2.shape)
    targetNames = np.unique(Y2)
    le          = LabelEncoder()
    Y2      = le.fit_transform(Y2)
    scaler2            = MinMaxScaler(feature_range=(0, 1))
    X2 = scaler2.fit_transform(X2)
    print(Y2)
    print(X2.shape)
    print(Y2.shape)
    X3 = []
    Y3 = []
    for i in range(len(ds3)):
        image = ds3[i]
        hu = fd_hu_moments(image)
        haralick   = fd_haralick(image)
        histogram  = fd_histogram(image)
        global_feature = np.hstack([histogram, haralick, hu])
        Y3.append(ds3_label[i])
        X3.append(global_feature)
    Y3 = np.asarray(Y3)
    X3 = np.asarray(X3)
    print(X3.shape)
    print(Y3.shape)
    targetNames = np.unique(Y3)
    le          = LabelEncoder()
    Y3      = le.fit_transform(Y3)
    scaler3            = MinMaxScaler(feature_range=(0, 1))
    X3 = scaler3.fit_transform(X3)
    print(Y3)
    print(X3.shape)
    print(Y3.shape)
    text.insert(END,filename+' dataset loaded\n')
    text.insert(END,"Total images found in dataset is : "+str(len(X1)+len(X2)+len(X3))+"\n")

def runSVM():
    text.delete('1.0', END)
    global svm1, svm2, svm3
    global X_train1, X_test1, y_train1, y_test1
    global X_train2, X_test2, y_train2, y_test2
    global X_train3, X_test3, y_train3, y_test3
    global X1, Y1
    global X2, Y2
    global X3, Y3
    global svm_graph
    global knn_graph
    global lr_graph
    global elm_graph
    elm_graph.clear()
    svm_graph.clear()
    knn_graph.clear()
    lr_graph.clear()

    indices = np.arange(X1.shape[0])
    np.random.shuffle(indices)
    X1 = X1[indices]
    Y1 = Y1[indices]
    indices = np.arange(X2.shape[0])
    np.random.shuffle(indices)
    X2 = X2[indices]
    Y2 = Y2[indices]
    indices = np.arange(X3.shape[0])
    np.random.shuffle(indices)
    X3 = X3[indices]
    Y3 = Y3[indices]

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size = 0.10)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size = 0.10)
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, Y3, test_size = 0.10)

    svm1 = SVC()
    svm1.fit(X_train1,y_train1)
    predict = svm1.predict(X_test1);
    y_pred = accuracy_score(y_test1,predict)
    svm_graph.append(y_pred)
    text.insert(END,"SVM RGB Classification Rate : "+str(y_pred)+"\n")
    svm2 = SVC()
    svm2.fit(X_train2,y_train2)
    predict = svm2.predict(X_test2);
    y_pred = accuracy_score(y_test2,predict)
    svm_graph.append(y_pred)
    text.insert(END,"SVM YCBCR Classification Rate : "+str(y_pred)+"\n")
    svm3 = SVC()
    svm3.fit(X_train3,y_train3)
    predict = svm3.predict(X_test3);
    y_pred = accuracy_score(y_test3,predict)
    svm_graph.append(y_pred)
    text.insert(END,"SVM HSV Classification Rate : "+str(y_pred)+"\n\n")

def runKNN():
    global X1, Y1
    global X2, Y2
    global X3, Y3
    global X_train1, X_test1, y_train1, y_test1
    global X_train2, X_test2, y_train2, y_test2
    global X_train3, X_test3, y_train3, y_test3
    knn1 = KNeighborsClassifier(n_neighbors = 7)
    knn1.fit(X1,Y1)
    predict = knn1.predict(X_test1);
    y_pred = accuracy_score(y_test1,predict)
    knn_graph.append(y_pred)
    text.insert(END,"KNN RGB Classification Rate : "+str(y_pred)+"\n")
    knn2 = KNeighborsClassifier(n_neighbors = 9)
    knn2.fit(X2,Y2)
    predict = knn2.predict(X_test2);
    y_pred = accuracy_score(y_test2,predict)
    knn_graph.append(y_pred)
    text.insert(END,"KNN YCBCR Classification Rate : "+str(y_pred)+"\n")
    knn3 = KNeighborsClassifier(n_neighbors = 2)
    knn3.fit(X3,Y3)
    predict = knn3.predict(X_test3);
    y_pred = accuracy_score(y_test3,predict)
    knn_graph.append(y_pred)
    text.insert(END,"KNN HSV Classification Rate : "+str(y_pred)+"\n\n")

def runLR():
    global X1, Y1
    global X2, Y2
    global X3, Y3
    global X_train1, X_test1, y_train1, y_test1
    global X_train2, X_test2, y_train2, y_test2
    global X_train3, X_test3, y_train3, y_test3
    lr1 = LogisticRegression()
    lr1.fit(X_train1,y_train1)
    predict = lr1.predict(X_test1);
    y_pred = accuracy_score(y_test1,predict)
    lr_graph.append(y_pred)
    text.insert(END,"Logistic Regression RGB Classification Rate : "+str(y_pred)+"\n")
    lr2 = LogisticRegression()
    lr2.fit(X_train2,y_train2)
    predict = lr2.predict(X_test2);
    y_pred = accuracy_score(y_test2,predict)
    lr_graph.append(y_pred)
    text.insert(END,"Logistic Regression YCBCR Classification Rate : "+str(y_pred)+"\n")
    lr3 = LogisticRegression()
    lr3.fit(X_train3,y_train3)
    predict = lr3.predict(X_test3);
    y_pred = accuracy_score(y_test3,predict)
    lr_graph.append(y_pred)
    text.insert(END,"Logistic Regression HSV Classification Rate : "+str(y_pred)+"\n\n")

def runELM():
    global X1, Y1
    global X2, Y2
    global X3, Y3
    global X_train1, X_test1, y_train1, y_test1
    global X_train2, X_test2, y_train2, y_test2
    global X_train3, X_test3, y_train3, y_test3
    srhl_tanh = MLPRandomLayer(n_hidden=500, activation_func='tanh')
    elm1 = GenELMClassifier(hidden_layer=srhl_tanh)
    elm1.fit(X1,Y1)
    predict = elm1.predict(X_test1);
    y_pred = accuracy_score(y_test1,predict)
    elm_graph.append(y_pred)
    text.insert(END,"Extension Extreme Learning Machine RGB Classification Rate : "+str(y_pred)+"\n")
    elm2 = GenELMClassifier(hidden_layer=srhl_tanh)
    elm2.fit(X2,Y2)
    predict = elm2.predict(X_test2);
    y_pred = accuracy_score(y_test2,predict)
    elm_graph.append(y_pred)
    text.insert(END,"Extension Extreme Learning Machine YCBCR Classification Rate : "+str(y_pred)+"\n")
    elm3 = GenELMClassifier(hidden_layer=srhl_tanh)
    elm3.fit(X3,Y3)
    predict = elm3.predict(X_test3);
    y_pred = accuracy_score(y_test3,predict)
    elm_graph.append(y_pred)
    text.insert(END,"Extension Extreme Learning Machine HSV Classification Rate : "+str(y_pred)+"\n\n")
    

def uploadTestImage():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "testImages")
    text.insert(END,filename+" test image loaded")
    
def predict():
    img = cv2.imread(filename)
    img = cv2.resize(img,(64,64))
    hu = fd_hu_moments(img)
    haralick   = fd_haralick(img)
    histogram  = fd_histogram(img)
    img = np.hstack([histogram, haralick, hu])
    img = np.asarray(img)
    temp1 = scaler1.transform([img])
    temp2 = scaler2.transform([img])
    temp3 = scaler3.transform([img])
    harvest_time = svm1.predict(temp1)
    growth_rate = svm2.predict(temp2)
    phenology = svm3.predict(temp3)
    types = ''
    if phenology[0] == 0:
        types = 'CabernetSauvignon'
    if phenology[0] == 1:
        types = 'Shiraz'    
    img = cv2.imread(filename)
    img = cv2.resize(img,(600,600))
    cv2.putText(img, 'Harvest Time Merlot Round '+str(harvest_time[0]), (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 0), 2)
    cv2.putText(img, 'Growth Rate Round '+str(growth_rate[0]), (10, 45),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 0), 2)
    cv2.putText(img, 'Phenology = '+str(types), (10, 65),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 255, 0), 2)
    cv2.imshow('Classification Result', img)
    cv2.waitKey(0)
                

def graph():
    df = pd.DataFrame([['SVM','RGB Rate',svm_graph[0]],['SVM','YCBCR Rate',svm_graph[1]],['SVM','HSV Rate',svm_graph[2]],
                       ['KNN','RGB Rate',knn_graph[0]],['KNN','YCBCR Rate',knn_graph[1]],['KNN','HSV Rate',knn_graph[2]],
                     
                       ['Extension ELM','RGB Rate',elm_graph[0]],['Extension ELM','YCBCR Rate',elm_graph[1]],['Extension ELM','HSV Rate',elm_graph[2]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

    
font = ('times', 16, 'bold')
title = Label(main, text='Computer Vision and Machine Learning for Viticulture Technology')
title.config(bg='firebrick4', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="GrapeCS-ML Database", command=uploadDataset, bg='#ffb3fe')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM, bg='#ffb3fe')
svmButton.place(x=280,y=550)
svmButton.config(font=font1)

knnButton = Button(main, text="Run KNN Algorithm", command=runKNN, bg='#ffb3fe')
knnButton.place(x=480,y=550)
knnButton.config(font=font1)



elmButton = Button(main, text="Run Extension Extreme Learning Machine Algorithm", command=runELM, bg='#ffb3fe')
elmButton.place(x=50,y=600)
elmButton.config(font=font1)

testButton = Button(main, text="Upload Test Image", command=uploadTestImage, bg='#ffb3fe')
testButton.place(x=480,y=600)
testButton.config(font=font1) 

predictButton = Button(main, text="Predict Harvest Stage, Growth Stage and Phenology Type", command=predict, bg='#ffb3fe')
predictButton.place(x=680,y=600)
predictButton.config(font=font1) 

exitButton = Button(main, text="All Algorithms Comparison Graph", command=graph, bg='#ffb3fe')
exitButton.place(x=50,y=650)
exitButton.config(font=font1) 


main.config(bg='LightSalmon3')
main.mainloop()
