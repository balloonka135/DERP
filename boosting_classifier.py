#! /usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
import picture_iterator
import csv_merger
import data_reader


class ErrorEvaluator:
    def __init__(self):
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0

    def valuate(self, val_real, val_predict):
        if val_real:
            if val_predict:
                self.true_pos += 1
            else:
                self.false_neg += 1
        else:
            if val_predict:
                self.false_pos += 1
            else:
                self.true_neg += 1

    def precision(self):
        return float(self.true_pos) / (self.true_pos + self.false_pos)

    def recall(self):
        return float(self.true_pos) / (self.true_pos + self.false_neg)

    def f_measure(self):
        return 2*(self.precision()*self.recall())/(self.precision() + self.recall())


def train_classifier(path_to_classifier_xml="Data/boost_emotions_classifier.xml",
                     data_path='Data/dataset/merged_data_train.csv',
                     emotion=1):
    data = []
    with open(data_path, 'r') as data_file:
        for data_line in data_file:
            data.append([float(value) for value in data_line.split(',')])
    data = np.float32(np.array(data))

    smile_responses = [1 if value == emotion else 0 for value in data[:, 1]]
    smile_responses = np.float32(np.array(smile_responses))

    params = dict(max_depth=5,
                  boost_type=cv2.BOOST_LOGIT,
                  weak_count=300,
                  #weight_trim_rate=0.01,
                  use_surrogates=True,
                  priors=0
                  )
    booster = cv2.Boost(trainData=data[:, 2:], tflag=cv2.CV_ROW_SAMPLE, responses=smile_responses, params=params)
    booster.train(data[:, 2:], tflag=cv2.CV_ROW_SAMPLE, responses=smile_responses, params=params)

    booster.save(path_to_classifier_xml)


def test_classifier(path_to_classifier_xml="Data/boost_emotions_classifier.xml",
                    path_to_data="Data/dataset/merged_data_val.csv",
                    emotion=1):
    classifier = cv2.Boost()
    classifier.load(path_to_classifier_xml)

    err_eval = ErrorEvaluator()
    d_reader = data_reader.DataReader(path_to_data)
    for line_num, data_line in enumerate(d_reader.get_objects()):
        emotion_real = 1 if d_reader.get_classes()[line_num] == emotion else 0
        emotion_predicted = classifier.predict(np.float32(np.array(data_line)))
        err_eval.valuate(emotion_real, emotion_predicted)

    print("total positives "),
    print(err_eval.true_pos + err_eval.false_neg)
    print("Precision: %s" % err_eval.precision())
    print("Recall: %s" % err_eval.recall())
    print("F-measure: %s" % err_eval.f_measure())
    return err_eval


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
    train_classifier(emotion=1)
    test_classifier(emotion = 1)
