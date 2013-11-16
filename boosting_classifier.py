#! /usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
import csvviewer
import csv_merger

def train(data_path='Data/merged_data.csv', emotion=1):
    data = []
    with open(data_path,'r') as data_file:
        for data_line in data_file:
            data.append([float(value) for value in data_line.split(',')])
    data = np.float32(np.array(data))
    smile_responses = [1 if value == emotion else 0 for value in data[:,1]]
    smile_responses = np.float32(np.array(smile_responses))
    print data[:,1].shape
    booster = cv2.Boost(trainData=data[:, 2:],tflag=cv2.CV_ROW_SAMPLE, responses=smile_responses)
    booster.train(data[:,2:],tflag=cv2.CV_ROW_SAMPLE, responses=smile_responses)
    booster.save("Data/boost_emotions_classifier.xml")


def classify(path_to_classifier_xml = "Data/boost_emotions_classifier.xml",show_only_good = False):
    booster = cv2.Boost()
    booster.load(path_to_classifier_xml)

    picture_collection = csvviewer.picture_collection(path='../training.csv')
    dots = []
    pic_iterator = csvviewer.picture_dot_iterator(path='../training.csv',dots=dots)
    for pic in pic_iterator:
        distances  = csv_merger.count_distances(pic,dots)
        distances = np.float32(np.array(distances))
        result = booster.predict(distances)
        copy = np.zeros((96, 192), dtype=np.uint8)
        copy[:, :96] = picture_collection[int(pic['number'])]['Image'][:,:]
        if not show_only_good :
            if result < 1:
                continue
        cv2.putText(img=copy,
                    text="found" if result > 0 else "not found",
                    org=(96, 48),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.8,
                    color=(255, 255, 255))
        cv2.imshow('smile',copy)
        cv2.waitKey(-1)



if __name__ == "__main__":
    train(emotion=4)
    classify()