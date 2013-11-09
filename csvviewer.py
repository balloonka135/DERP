#! /usr/bin/env python
# coding=utf-8

import csv
import cv2
import numpy as np

import config


def picture_iterator(path='training.csv', start_at=0, end_at=-1):
    with open(path, 'rb') as pictures_file:
        pictures = csv.DictReader(pictures_file)
        for i, picture in enumerate(pictures):
            if i < start_at:
                continue

            if end_at != -1 and i > end_at:
                raise StopIteration()

            image = [int(x) for x in picture['Image'].split(' ')]
            picture['Image'] = np.uint8(np.array(image).reshape((96, 96)))
            yield picture

if __name__ == "__main__":
    window = 'emotion tag'
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    with open('result.csv', 'ab') as result_file:
        dw = csv.DictWriter(result_file, ('emotion', 'number'))

        emotions = [('Страх', 0),
                    ('Радость', 1),
                    ('Печаль', 2),
                    ('Гнев', 3),
                    ('Удивление', 4),
                    ('Отвращение', 5),
                    ('Удовольствие', 6),
                    ('Нейтральный', 7)]

        for em, key in emotions:
            print em, ' => ', key

        start = config.START
        end = config.END
        pic_num = start
        for pic in picture_iterator(start_at=start, end_at=end):
            copy = np.zeros((96, 192), dtype=np.uint8)
            copy[:, :96] = pic['Image'][:, :]

            cv2.putText(img=copy,
                        text=str(pic_num),
                        org=(96, 48),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.8,
                        color=(255, 255, 255))

            cv2.imshow(window, copy)

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