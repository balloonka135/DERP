#! /usr/bin/env python
# coding=utf-8

import csv
import cv2
import numpy as np
import config
import picture_iterator as p_iter

import collections
EMOTIONS = {'Страх': 0,
            'Радость': 1,
            'Печаль': 2,
            'Гнев': 3,
            'Удивление': 4,
            'Отвращение': 5,
            'Удовольствие': 6,
            'Нейтральный': 7}
EMOTION_KEYS = {config.KEY_CODE_0: EMOTIONS['Страх'],#ord('0'): 0, #1 - страх
                config.KEY_CODE_1: EMOTIONS['Радость'],#ord('1'): 1, #2 - радость
                config.KEY_CODE_2: EMOTIONS['Печаль'],#ord('2'): 2, #3 - печаль
                config.KEY_CODE_3: EMOTIONS['Гнев'],#ord('3'): 3, #4 - гнев
                config.KEY_CODE_4: EMOTIONS['Удивление'],#ord('4'): 4, #5 - удивление
                config.KEY_CODE_5: EMOTIONS['Отвращение'],#ord('5'): 5, #6 - отвращение
                config.KEY_CODE_6: EMOTIONS['Удовольствие'],#ord('6'): 6, #7 - удовольствие
                config.KEY_CODE_7: EMOTIONS['Нейтральный'] #ord('7'): 7  #8 - нейтральный
                }


if __name__ == "__main__":
    window = 'emotion tag'
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    for i in collections.KeysView(EMOTIONS):
        print (i)



    start = config.START
    end = config.END

    picture_iterator = p_iter.PictureCollection(start=start, end=end)
    with open('result.csv', 'ab') as result_file:
        dw = csv.DictWriter(result_file, ('emotion', 'number'))
        for pic in picture_iterator:
            copy = np.zeros((96, 192), dtype=np.uint8)
            copy[:, :96] = pic['Image'][:, :]

            cv2.putText(img=copy,
                        text=str(pic['number']),
                        org=(96, 48),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.8,
                        color=(255, 255, 255))
            cv2.imshow(window, copy)
            descriptor = {'number': pic['number']}
            ok_key = False
            while not ok_key:
                c = cv2.waitKey(-1)
                if c == config.KEY_CODE_ESC:
                    break
                try:
                    descriptor['emotion'] = EMOTION_KEYS[c]
                except KeyError:
                    continue
                ok_key = True
            if not ok_key:  # escaped
                break
            dw.writerow(descriptor)