from __future__ import print_function
import tkinter as tk from tkinter import * #import tkMessageBox import tkinter.messagebox as messagebox from tkinter import filedialog
# keras imports from keras.applications.vgg16 import VGG16, preprocess_input from keras.applications.vgg19 import VGG19, preprocess_input from keras.applications.xception import Xception, preprocess_input from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input from keras.applications.mobilenet import MobileNet, preprocess_input from keras.applications.inception_v3 import InceptionV3, preprocess_input from keras.preprocessing import image from keras.models import Model from keras.models import model_from_json
from keras.layers import Input
import math
from sklearn.ensemble import VotingClassifier from imutils import paths
# other imports
from sklearn.linear_model import LogisticRegression import numpy as np import os import json import pickle import cv2
# load the user configs with open('conf/conf.json') as f:    config = json.load(f)
# initializing tkinter top = tk.Tk() # config variables model_name  = config["model"] weights  = config["weights"] include_top  = config["include_top"] train_path  = config["train_path"] test_path  = config["test_path"] features_path  = config["features_path"] labels_path  = config["labels_path"] test_size  = config["test_size"] results  = config["results"] model_path  = config["model_path"] seed  = config["seed"] classifier_path = config["classifier_path"]
# load the trained logistic regression classifier print ("[INFO] loading the classifier...")
classifier1 = pickle.load(open("output\\skin\\inceptionv3\\classifier.pickle", 'rb')) classifier2 = pickle.load(open("output\\skin\\inceptionresnetv2\\classifier.pickle", 'rb')) classifier3 = pickle.load(open("output\\skin\\mobilenet\\classifier.pickle", 'rb'))
model_name1= "inceptionv3"
base_model1 = InceptionV3(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
model1 = Model(input=base_model1.input, output=base_model1.get_layer('custom').output) image_size1 = (299, 299) model_name2 = "inceptionresnetv2"
base_model2 = InceptionResNetV2(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
model2 = Model(input=base_model2.input, output=base_model2.get_layer('custom').output) image_size2 = (299, 299)
model_name3 = "mobilenet" base_model3 = MobileNet(include_top=include_top, weights=weights, input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
model3 = Model(input=base_model3.input, output=base_model3.get_layer('custom').output) image_size3 = (224, 224)
# Chime = "input.wav"
def helloCallBack(): filename = filedialog.askopenfilename()
# get all the train labels train_labels = os.listdir(train_path)
# get all the test images paths #test_images = os.listdir(test_path) labelss=[] res=[] cnt=0
p = []
i = 0 imagePath=filename
#for imagePath in paths.list_images("dataset\\test"): i = i+1
#path  = test_path + "/" + image_path print(imagePath)
# make = imagePath.split("\\")[-2]
# print (make) pred=[] img1  = image.load_img(imagePath, target_size=image_size1) img2  = image.load_img(imagePath, target_size=image_size3) x1  = image.img_to_array(img1) x1  = np.expand_dims(x1, axis=0) x1  = preprocess_input(x1) feature1  = model1.predict(x1) feature2  = model2.predict(x1) flat1  = feature1.flatten() flat1  = np.expand_dims(flat1, axis=0) preds1  = classifier1.predict(flat1) pred.append(preds1[0]) prediction1 = train_labels[preds1[0]] flat2  = feature2.flatten() flat2  = np.expand_dims(flat2, axis=0) preds2  = classifier2.predict(flat2) pred.append(preds2[0]) prediction2 = train_labels[preds2[0]] x2  = image.img_to_array(img2) x2  = np.expand_dims(x2, axis=0) x2  = preprocess_input(x2) feature3  = model3.predict(x2) flat3  = feature3.flatten() flat3  = np.expand_dims(flat3, axis=0) preds3  = classifier3.predict(flat3) prediction3 = train_labels[preds3[0]] pred.append(preds3[0]) idx, ctr = 0, 1 for i in range(1, len(pred)): if pred[idx] == pred[i]: ctr += 1 else: ctr -= 1 if ctr == 0: idx = 1 ctr = 1
#fin=np.average(int(pred),axis=0) fin_prediction = train_labels[pred[idx]] print("pred1",prediction1) print("pred2",prediction2) print("pred3",prediction3) print("final",fin_prediction) # if make==fin_prediction:
#  cnt=cnt+1
#  print (cnt)
# perform prediction on test image print ("I think it is a " + fin_prediction) img_color = cv2.imread(imagePath, 1)
cv2.putText(img_color, "I think it is a " + fin_prediction, (140,445), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) cv2.imshow("test", img_color)
# key tracker key = cv2.waitKey(1000) 
messagebox.showinfo(message=("Disease - "+fin_prediction))
#  p= paths.list_images("dataset\\test")
# acc=cnt/i # print(acc*100)
cv2.destroyAllWindows()
browsebutton = tk.Button(top, text="Predict_Disease", command=helloCallBack) browsebutton.pack() top.mainloop() 
