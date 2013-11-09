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
    """
    for pic in picture_iterator():
        for dot in pic.keys():
            if dot != 'Image':
                cv2.circle(pic['Image'], (int(float(pic[ dot[:-1]+'x'])),int(float(pic[ dot[:-1]+'y']))),2,(255,) )
        cv2.imshow('img',pic['Image'])
        cv2.waitKey(1000)
    """
    with open('result.csv','a') as result_file:
        dw = csv.DictWriter(result_file, ('emotion', 'number'))


        emotions = {'Страх': 0,
                    'Радость': 1,
                    'Печаль': 2,
                    'Гнев': 3,
                    'Удивление': 4,
                    'Отвращение': 5,
                    'Удовольствие': 6,
                    'Нейтральный': 7}

        for em in emotions:
            print em,' => ', emotions[em]

        pic_num = 3001
        max_pic_num = 4000
        for pic in picture_iterator(start_at=pic_num):
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
            if pic_num == max_pic_num:
                break
            pic_num += 1
