#! /usr/bin/env python
# coding=utf-8

import csv
import cv2
import numpy as np


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

        start = 621
        end = 2000
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
            if c == 1048603:
                break

            descriptor['emotion'] = {
                1048625: 0,#ord('0'): 0, #1 - страх
                1048626: 1,#ord('1'): 1, #2 - радость
                1048627: 2,#ord('2'): 2, #3 - печаль
                1048628: 3,#ord('3'): 3, #4 - гнев
                1048629: 4,#ord('4'): 4, #5 - удивление
                1048630: 5,#ord('5'): 5, #6 - отвращение
                1048631: 6,#ord('6'): 6, #7 - удовольствие
                1048632: 7 #ord('7'): 7  #8 - нейтральный
            }[c]

            dw.writerow(descriptor)
            pic_num += 1
