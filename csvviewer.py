#! /usr/bin/env python
# coding=utf-8

import csv
import cv2
import numpy as np

import config


class picture_collection:
    def __init__(self,path='training.csv',):
        self.collection = []
        self.path = path

    def __getitem__(self, item):
        if (item < len(self.collection)):
            return self.collection[item]
        else:
            start = len(self.collection)
            for pic in picture_iterator(self.path, start, item+1):
                self.collection.append(pic)
            return self.collection[item]

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

def picture_dot_iterator(path='training.csv', start_at=0, end_at=-1, only_full=True, dots= None):
    with open(path, 'rb') as pictures_file:
        pictures = csv.DictReader(pictures_file)

        if dots != None:
            all_dots = list(pictures.fieldnames)
            del all_dots[all_dots.index('Image')]
            dots.extend([ (dot1,dot2) for dot1 in all_dots for dot2 in all_dots
                          if dot2[:-1]==dot1[:-1] and dot1[-1] == 'x' and dot2[-1]=='y' ])
        for i, picture in enumerate(pictures):
            if i < start_at:
                continue

            if end_at != -1 and i > end_at:
                raise StopIteration()

            if only_full:
                if '' in picture.values():
                    continue
            del picture['Image']
            picture['number'] = str(i)
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
            ok_key = False
            while not ok_key:
                c = cv2.waitKey(-1)
                if c == config.KEY_CODE_ESC:
                    break
                try:
                    descriptor['emotion'] = {
                        config.KEY_CODE_0: 0,#ord('0'): 0, #1 - страх
                        config.KEY_CODE_1: 1,#ord('1'): 1, #2 - радость
                        config.KEY_CODE_2: 2,#ord('2'): 2, #3 - печаль
                        config.KEY_CODE_3: 3,#ord('3'): 3, #4 - гнев
                        config.KEY_CODE_4: 4,#ord('4'): 4, #5 - удивление
                        config.KEY_CODE_5: 5,#ord('5'): 5, #6 - отвращение
                        config.KEY_CODE_6: 6,#ord('6'): 6, #7 - удовольствие
                        config.KEY_CODE_7: 7 #ord('7'): 7  #8 - нейтральный
                    }[c]
                except KeyError:
                    continue
                ok_key = True
            if not ok_key:  # escaped
                break
            dw.writerow(descriptor)
            pic_num += 1
