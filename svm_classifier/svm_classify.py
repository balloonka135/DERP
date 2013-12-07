#!/usr/bin/python
#coding=utf-8

import cv2
from sklearn import svm
import numpy
import cPickle as cpkl
import config
from data_reader import DataReader

input_clf = open(config.SVM_CLF_DIR+"/trained_classifier.pkl", "rb")
clf = cpkl.load(input_clf)
reader = DataReader(config.TEST_DATA_PATH)
test_objects = reader.get_objects()
test_classes = reader.get_classes()

for i in range(len(test_objects)):
    print("real class:%d, predicted class:%d\n"% test_classes[i], clf.predict(test_objects[i]))

input_clf.close()
input_obj.close()
input_cls.close()
