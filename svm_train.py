#!/usr/bin/python
#coding=utf-8

import csv
import cv2
from sklearn import svm
from sklearn import cross_validation as cv
import numpy
import cPickle as cpkl
import os
import shutil
import glob

FEARNESS = 0 #страх
JOYNESS = 1 #радость
MADNESS = 2 #печаль
ANGER = 3 #гнев
SURPRISE = 4 #удивление
DISGUST = 5 #отвращение
PLEASURE = 6 #удовольствие
NEUTRAL = 7 #нейтральный

# В качестве объекта берутся расстояния между ключевыми точками на изображении
#Классы от 0 до 7
classes = []
objects = []
## Данные для тестирования
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

## Далее применим 3-way-cross-validation (60-20-20)
#Читаем данные для тренировки
train_data_reader = DataMaker(train_data_path)
classes = train_data_reader.get_classes()
objects = train_data_reader.get_objects()

#тренируем классификатор
classifier = svm.SVC(kernel='linear')
classifier.fit(objects, classes)

classes = list()
objects = list()
#читаем данные для кросс валидации
val_data_reader = DataMaker(val_data_path)
classes = val_data_reader.get_classes()
objects = val_data_reader.get_objects()

####
######### здесь будет cross - validation tests
####

#читаем данные для тестирования
#(далее передадим их в качестве объектов для файла svm_classify.py)
test_data_reader = DataMaker(test_data_path)
test_classes = test_data_reader.get_classes()
test_objects = test_data_reader.get_objects()

#сохраняем тренированный объект-классификатор в файл
#а также сохраняем объекты на которых будем тестировать классификатор
#еще для сравнения сохраняем объекты

newpath = r"./Data/trained_data_svm/"
if not os.path.exists(newpath):
    os.makedirs(newpath)

output_clf = open(r"./Data/trained_data_svm/trained_classifier.pkl", "wb")
output_obj = open(r"./Data/trained_data_svm/objects.pkl", "wb")
output_cls = open(r"./Data/trained_data_svm/classes.pkl", "wb")

cpkl.dump(classifier, output_clf)
cpkl.dump(test_objects, output_obj)
cpkl.dump(test_classes, output_cls)

output_clf.close()
output_obj.close()
output_cls.close()


