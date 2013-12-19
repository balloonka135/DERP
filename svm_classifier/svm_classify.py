#!/usr/bin/python
#coding=utf-8

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
from DERP import picture_iterator # for iteration and show picture


class SVMClassifier(object):
    def __init__(self): 
        ## Из файла - объекта принимаем классификатор
        self.input_ = open(os.path.join(config.SVM_CLF_DIR,"trained_classifier.pkl"), "rb")
        self.clf = cpkl.load(self.input_)
        
    def __del__(self):
        self.input_.close()

    ##Метод show используется методом predict для отображения данных
    ##Параметры по умолчанию переданы для возможности отображения тестовых данных
    def __show(self, predicted_class, pic=None, pictures=None):
        image = np.zeros((96, 192), dtype=np.uint8)
        image[:, :96] = pictures[ int(pic['number']) ]['Image'][:, :] 

        text_predicted = {
                config.FEAR:"Fear",
                config.JOY:"Joy",
                config.SADNESS:"Sadness",
                config.ANGER:"Anger",
                config.SURPRISE:"Surprise",
                config.DISGUST:"Disgust",
                config.PLEASURE:"Pleasure",
                config.NEUTRAL:"Neutral",
                }.get(predicted_class[0], "Neutral")


        ##Скорее всего нужно будет немного поменять для видеофрейма...
        cv2.putText(img=image,
                    text=text_predicted,
                    org=(96,48),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1.5,
                    color=(255,255,255))
        '''
        cv2.putText(img=image,
                    text=str(pic['number']),
                    org=(96,72),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=0.9,
                    color=(255,255,255))
        '''

        cv2.imshow("DERP", image)
        c = cv2.waitKey(-1)
        if c == config.KEY_CODE_ESC:
            raise SystemExit
    
    ## Метод для рассчета расстояний, используется csv_merger.count_distances
    def get_distances(self, keypoints, norm=None):
        self.keypoints = keypoints
        object_ = csv_merger.count_distances(self.keypoints, norm)
        return object_ 
    
    ## Метод для парсинга ключевых точек тестовых данных
    ## Возможно пригодиться для реальных данных с видео фрейма
    def parse_keypoints(self, pic, knames):
        keypoints = list()
        for dot_type_x, dot_type_y in knames:
            keypoints.append((pic[dot_type_x], pic[dot_type_y]))
        return keypoints
            
    ## Метод для предсказания эмоции, передаются iterable pictures и картинка pic 
    ## Необязательные парaметры необходимы для тестовых данных
    def predict(self, object_, pic=None, pictures=None):
        pred_class = self.clf.predict(object_)
        self.__show(pred_class, pic, pictures)
        return pred_class

if __name__ == '__main__':
    
    classifier = SVMClassifier()

    ####### тестовые данные для интерактивного просмотра
    ## Получаем изображения из файла training.csv и значения типов точек для изображений

    pictures = picture_iterator.PictureCollection(path="../../training.csv")
    keypoint_names = pictures.key_points

    for pic in pictures:
        kpoints = classifier.parse_keypoints(pic, keypoint_names)
        object_ = classifier.get_distances(kpoints, csv_merger.norm(pic))
        class_ = classifier.predict(object_,
                                   pic=pic,
                                   pictures=pictures)
