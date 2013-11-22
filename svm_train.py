#!/usr/bin/python
#coding=utf-8

import csv
import cv2
from sklearn import svm
import numpy
import cPickle as cpkl
import os
import shutil
import glob

# В качестве объекта берутся расстояния между ключевыми точками на изображении
objects=[]
#Классы от 0 до 7
classes=[]

FEARNESS = 0 #страх
JOYNESS = 1 #радость
MADNESS = 2 #печаль
ANGER = 3 #гнев
SURPRISE = 4 #удивление
DISGUST = 5 #отвращение
PLEASURE = 6 #удовольствие
NEUTRAL = 7 #нейтральный

temp=[]
#парсинг файла для тренировки классификатора
with open('merged_data.csv', 'rb') as fd:
	#читаем данные в список(отдельный список для каждого изображения):
	#['номер картинки', 'эмоция', 'расстояние1',...,'расстояние105']
	training_data = csv.reader(fd)
	#бежим по каждому списку
	for data in training_data:
		#в список classes кладем эмоции 
		classes.append(int(data[1]))
		#бежим по расстояниям и кладем их в отдельный список
		#далее мы будем тренировать классфикатор, по данным объектам
		#элементу из списка classes соответствует элемент-список из списка objects
		for i in range(2,107):
			temp.append(float(data[i]))
		objects.append(temp)
		temp=list()

#данные получены, теперь можно делать с ними все захочется
#будем выполнять бинарную классификацию
#классификатор будет определять эмоцию радость на изображении
#сделаем cross-validation

class_joyness_or_not=[1 if i==JOYNESS  else 0 for i in classes]
#заводим объект-классификатор SVM
classifier = svm.LinearSVC()
#95% данных для тренировки (2033)
crossval_datalen = int(len(objects)*95/100)
#5% оставшихся данных для тестирования (107)
test_datalen = len(objects)-crossval_datalen   
#тренируем классификатор на 95% данных
#пока что просто fit, далее надо читать про мультиклассовую классификацию
classifier.fit(objects[:crossval_datalen], class_joyness_or_not[:crossval_datalen])

#сохраняем тренированный объект-классификатор в файл
#и объекты чтобы не считывать снова csv - файл а сразу провести тест
#на имеющихся данных
newpath = r"./trained_data/"
if not os.path.exists(newpath):
	os.makedirs(newpath)
output_clf = open(r"./trained_data/trained_classifier.pkl","wb")
output_obj = open(r"./trained_data/objects.pkl","wb")
cpkl.dump(classifier, output_clf)
cpkl.dump(objects, output_obj)
output_clf.close()
output_obj.close()



