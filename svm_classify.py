#!/usr/bin/python
#coding=utf-8

import cv2
import csv
from sklearn import svm
import numpy
import cPickle as cpkl

input_clf = open("./Data/trained_data/trained_classifier.pkl", "rb")
input_obj = open("./Data/trained_data/objects.pkl", "rb")
input_cls = open("./Data/trained_data/classes.pkl", "rb")
classifier = cpkl.load(input_clf)
objects = cpkl.load(input_obj)
classes = cpkl.load(input_cls)

#dec_func = classifier.decision_function(objects[2100])
#print("predicted data:{0}".format(classifier.predict(objects[2104])))
#print("real emotion: {0}".format(classes[2104]))

result = classifier.predict(objects[2104])

'''
if result == 1:
	print "joyness"
else:
	print "not joyness"
'''

input_clf.close()
input_obj.close()
input_cls.close()
