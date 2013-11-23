#!/usr/bin/python
#coding=utf-8

import cv2
import csv
from sklearn import svm
import numpy
import cPickle as cpkl

input_clf = open("./trained_data/trained_classifier.pkl","rb")
input_obj = open("./trained_data/objects.pkl","rb")
classifier = cpkl.load(input_clf)
objects = cpkl.load(input_obj)
#todo :сделать классификацию последних 5% объектов и красиво вывести 
#картинка : эмоция 
result = classifier.predict(objects[2104])

if result == 1:
	print "joyness"
else:
	print "not joyness"

input_clf.close()
input_obj.close()
