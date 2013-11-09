#! /usr/bin/env python
# coding=utf-8

import csv
import cv2
import numpy as np


def picture_iterator(path='training.csv', start_at = 0):
    with open(path, 'rb') as pictures_file:
        pictures = csv.DictReader(pictures_file)
        for i, picture in enumerate(pictures):
            if i < start_at:
                continue

            image = [int(x) for x in picture['Image'].split(' ')]
            picture['Image'] = np.uint8(np.array(image).reshape((96, 96)))
            yield picture

if __name__ == "__main__":
    with open('result.csv', 'a') as result_file:
        dw = csv.DictWriter(result_file, ('emotion', 'number'))
        pic_num = 0

        emotions = {'Страх': 0,
                    'Радость': 1,
                    'Печаль': 2,
                    'Гнев': 3,
                    'Удивление': 4,
                    'Отвращение': 5,
                    'Удовольствие': 6,
                    'Нейтральный': 7}

        for em in emotions:
            print em, ' => ', emotions[em]

        for pic in picture_iterator(start_at=0):

            cv2.imshow('e2', pic['Image'])
            descriptor = {'number': pic_num}
            c = cv2.waitKey(-1)
            if c == 27:
                break

            descriptor['emotion'] = {
                ord('0'): 0, #0 - страх
                ord('1'): 1, #1 - радость
                ord('2'): 2, #2 - печаль
                ord('3'): 3, #3 - гнев
                ord('4'): 4, #4 - удивление
                ord('5'): 5, #5 - отвращение
                ord('6'): 6, #6 - удовольствие
                ord('7'): 7  #7 - нейтральный
            }[c]

            dw.writerow(descriptor)
            pic_num += 1
