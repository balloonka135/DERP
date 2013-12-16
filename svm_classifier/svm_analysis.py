#!/usr/bin/python

import cv2
from sklearn import svm
import numpy as np
import cPickle as cpkl

import os, sys
file_dir = os.path.dirname(__file__)
abs_path = os.path.abspath(file_dir)
sys.path.append(os.path.join(os.path.join(abs_path,".."),".."))

from DERP import data_reader
from DERP import config #pathes
from DERP import csv_merger # for distances calculation
from DERP import picture_iterator

if __name__=="__main__":

    ##classifier
    input_1 = open(os.path.join(config.SVM_CLF_DIR,"trained_classifier.pkl"), "rb")
    input_2 = open(os.path.join(config.SVM_CLF_DIR,"bin_trained_.pkl"), "rb")
    clf = cpkl.load(input_1)
    bin_clf = cpkl.load(input_2)
    input_1.close()
    input_2.close()
    #### test data
    reader = data_reader.DataReader(config.TEST_DATA_PATH)
    test_objects = reader.get_objects()
    test_classes = reader.get_classes()
    cls_bin  = [0 if i is not config.JOY else config.JOY for i in test_classes ]
    #####
    '''
    for i in range(len(test_objects)):
        print("real class:%d, predicted class:%d\n"%(test_classes[i], clf.predict(test_objects[i])))
    '''
    #####
    print(clf.score(test_objects,test_classes))
    print(bin_clf.score(test_objects,cls_bin))
