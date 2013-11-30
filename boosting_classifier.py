#! /usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
import picture_iterator
import csv_merger



def train(data_path='Data/dataset/merged_data_train.csv', emotion=1):
    data = []
    with open(data_path, 'r') as data_file:
        for data_line in data_file:
            data.append([float(value) for value in data_line.split(',')])
    data = np.float32(np.array(data))
    smile_responses = [1 if value == emotion else 0 for value in data[:, 1]]
    smile_responses = np.float32(np.array(smile_responses))
    print data[:, 1].shape
    params = dict(max_depth=10,
                  boost_type=cv2.BOOST_LOGIT,
                  weak_count=1000,
                  #weight_trim_rate=0.01,
                  use_surrogates=True,
                  priors=0
                  )
    booster = cv2.Boost(trainData=data[:, 2:], tflag=cv2.CV_ROW_SAMPLE, responses=smile_responses, params=params)
    booster.train(data[:, 2:], tflag=cv2.CV_ROW_SAMPLE, responses=smile_responses, params=params)

    booster.save("Data/boost_emotions_classifier.xml")


def classify(path_to_classifier_xml="Data/boost_emotions_classifier.xml",
             path_to_data="Data/dataset/merged_data_val.csv",
             emotion=1):
    params = dict(max_depth=10,
              boost_type=cv2.BOOST_REAL,
              weak_count=10,
              weight_trim_rate=0.01,
              use_surrogates=True,
              priors=0)
    classifier = cv2.Boost()
    classifier.load(path_to_classifier_xml)
    true_pos = 0.0
    true_neg = 0.0
    false_pos = 0.0
    false_neg = 0.0
    total_pic = 0.0
    total_pos = 0.0
    with open(path_to_data, 'r') as data_file:
        for data_line in data_file:
            data = [float(value) for value in data_line.split(',')]
            emotion_real = 1 if data[1] == emotion else 0

            emotion_predicted = classifier.predict(np.float32(np.array(data[2:])))
            total_pic += 1

            if emotion_real == 1:
                total_pos += 1
                if emotion_predicted > 0:
                    true_pos += 1
                else:
                    false_neg += 1
            else:
                if emotion_predicted > 0:
                    false_pos += 1
                else:
                    true_neg += 1
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / total_pos
    print ("total positives "),
    print (total_pos)
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F-measure: " + str(2*(precision*recall)/(precision+recall)))



def classify_and_show(path_to_classifier_xml="Data/boost_emotions_classifier.xml", show_only_good=False):
    booster = cv2.Boost()
    booster.load(path_to_classifier_xml)

    picture_collection = picture_iterator.PictureCollection()
    keypoints = picture_collection.key_points

    for pic in picture_collection:
        distances = csv_merger.count_distances(pic, keypoints)
        distances = np.float32(np.array(distances))
        result = booster.predict(distances)
        copy = np.zeros((96, 192), dtype=np.uint8)
        copy[:, :96] = picture_collection[int(pic['number'])]['Image'][:, :]
        if not show_only_good:
            if result < 1:
                continue
        cv2.putText(img=copy,
                    text="found" if result > 0 else "not found",
                    org=(96, 48),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.8,
                    color=(255, 255, 255))
        cv2.imshow('smile', copy)
        cv2.waitKey(-1)


if __name__ == "__main__":
    train(emotion=1)
    classify(emotion = 1)