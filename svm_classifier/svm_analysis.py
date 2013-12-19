#!/usr/bin/python

import cv2
from sklearn import svm
import numpy as np
import cPickle as cpkl
from sklearn.metrics import classification_report

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
    ## joy test classes
    test_bin_joy_cls = [0 if i is not config.JOY else config.JOY for i in test_classes ]

    #####
    print("multiclass classifier score:{0}%\n".format(
                            clf.score(test_objects,test_classes)*100))
    print("binary class classifier score(JOY Recognition):{0}%\n".format(
                            bin_clf.score(test_objects,test_bin_joy_cls)*100))
    real_classes, pred_classes = test_classes, clf.predict(test_objects)
    real_joy_classes, pred_joy_classes = test_bin_joy_cls, bin_clf.predict(test_objects)

    #writing analysis result to file
    with open(os.path.join(config.SVM_CLF_DIR,"svm_report.txt"),"w") as fd:
        fd.write("classification report about multi class classifier:\n")
        fd.write(classification_report(real_classes, pred_classes))
        fd.write("classification report about binary class classifier:\n")
        fd.write(classification_report(real_joy_classes, pred_joy_classes))

