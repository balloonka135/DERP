#!/usr/bin/python
#coding=utf-8

import csv
import cv2
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import cross_validation as val
import numpy
import cPickle as cpkl
import math
import matplotlib
import os
import shutil
import glob

FEARNESS = 0 #страх
JOYNESS = 1 #радость
MADNESS = 2 #печаль
ANGER = 3 #гнев
SURPRISE = 4 #удивлениe
DISGUST = 5 #отвращение
PLEASURE = 6 #удовольствие
NEUTRAL = 7 #нейтральный

# В качестве объекта берутся расстояния между ключевыми точками на изображении
#Классы от 0 до 7
classes = []
objects = []
# Данные для кроссвалидации
val_classes = []
val_objects = []
# Данные для тестирования
test_classes = []
test_objects = []

train_data_path = "./Data/dataset/merged_data_train.csv"
val_data_path = "./Data/dataset/merged_data_val.csv"
test_data_path = "./Data/dataset/merged_data_test.csv"

#читает данные в список вида:
#['номер картинки', 'эмоция', 'расстояние1',...,'расстояние105']
#в список classes кладем эмоции
#в список objects кладем объекты
#бежим по расстояниям и кладем их в список objects
#далее мы будем тренировать классфикатор, по данным объектам
#элементу из списка classes соответствует элемент-список из списка objects
class DataMaker(object):
    def __init__(self, path):
        self.path = path
        self.train_classes = []
        self.train_objects = []
        self.temp = []

    def get_classes(self):
        with open(self.path, 'rb') as fd:
            for data in fd:
                data = list(data.split(", "))
                self.train_classes.append(int(data[1]))
        return self.train_classes

    def get_objects(self):
        with open(self.path, 'rb') as fd:
            for data in fd:
                data = list(data.split(", "))
                for i in range(2, 107):
                    self.temp.append(float(data[i]))
                self.train_objects.append(self.temp)
                self.temp = list()
        return self.train_objects

##estimate classifier error
def evaluate_error(objects, classes, clf):
    err_count = 0
    for i in range(len(objects)):
        pr = clf.predict(objects[i])
        print("real:{0}\n predicted:{1}\n".format(classes[i], pr))
        if classes[i] != pr:
            err_count += 1
    err_percentage = (err_count*100)/len(objects)
    print("errors:{0}, labels:{1}\n".format(err_count, len(objects)))
    print("errors percentage:{0} %\n".format(err_percentage))

#Читаем все необходимые данные
train_data_reader = DataMaker(train_data_path)
classes = train_data_reader.get_classes()
objects = train_data_reader.get_objects()

cv_data_reader=DataMaker(val_data_path)
val_classes = cv_data_reader.get_classes()
val_objects = cv_data_reader.get_objects()

test_data_reader = DataMaker(test_data_path)
test_classes = test_data_reader.get_classes()
test_objects = test_data_reader.get_objects()

#for binary classification joy and non joy classes
#joy_cls = [1 if i == 1 else 0 for i in classes]


#clf = svm.LinearSVC()
#clf.fit(objects, classes)
#print("error percentage (underfitting check)\n:{0}%".format((1.0-clf.score(objects, classes))*100.))
#print("error percentage (overfitting check)\n:{0}%".format((1.0-clf.score(test_objects, test_classes))*100.))

# tuned parameters
C = [i for i in range(1, 10)]
intercept_scaling=[i for i in range (1, 3)]
tol = [0.001,0.0001]#,0.00001,0.00001]

parameters = [{'C':C,
             # 'loss':['l2'],
              #'penalty':['l2'],
              'class_weight':['auto',None],
              'intercept_scaling':intercept_scaling,
              'tol':tol,
              'dual':[False,True]}]

scores = ['precision']##['accuracy', 'average_precision', 'precision', 'recall', 'f1', 'roc_auc']

for score in scores:
    print("Tuning hyper-parameters for {0}\n".format(score))
    clf=GridSearchCV(svm.LinearSVC(), parameters, cv=3, scoring=score)
    clf.fit(objects, classes)

    print("Best parameters (with train set:\n")
    print(clf.best_estimator_)
    print("Grid scores (with train set):\n")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.5f (+/-%0.05f) for %r"
              % (mean_score, scores.std() / 2, params))
    print("Detailed classification report:\n")
    print("The model is trained on the full train set.\n")
    print("The scores are computed on the full validation set.\n")
    print("\n")
    real_classes, pred_classes = val_classes,clf.predict(val_objects)
    print(classification_report(real_classes, pred_classes))

