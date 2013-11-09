#! /usr/bin/env python
# coding=utf-8
import csv
import cv2
import numpy as np

def picture_iterator(path = './training.csv'):
    with open(path, 'rb') as pictures_file:
        pictures = csv.DictReader(pictures_file)
        for picture in pictures:

            image = [int(x) for x in picture['Image'].split(' ')]
            #print len(picture)
            picture['Image']= np.uint8(np.array(image).reshape((96,96)))
            yield  picture


    pass

if __name__ == "__main__":
    """
    for pic in picture_iterator():
        for dot in pic.keys():
            if dot != 'Image':
                cv2.circle(pic['Image'], (int(float(pic[ dot[:-1]+'x'])),int(float(pic[ dot[:-1]+'y']))),2,(255,) )
        cv2.imshow('img',pic['Image'])
        cv2.waitKey(1000)
    """
    with open('result.csv','w') as result_file:
        dw = csv.DictWriter(result_file, ('emotion', 'number'))
        pic_num = 0;

        emotions = {'страх': 0, 'радость': 1, 'печаль': 2, 'гнев': 3, 'Удивление':4 , 'Отвращение':5, 'удовольствие':6}
        for em in emotions:
            print em,' => ', emotions[em]
        for pic in picture_iterator():
            cv2.imshow('e2', pic['Image'])
            descriptor = {'number': pic_num}
            c = cv2.waitKey(-1)
            if c == 1048603:
                break
            descriptor['emotion'] = {
                1048625: 0, #1 - страх
                1048626: 1, #2 - радость
                1048627: 2, #3 - печаль
                1048628: 3, #4 - гнев
                1048629: 4, #5 - удивление
                1048630: 5, #6 - отвращение
                1048631: 6  #7 - удовольствие
            }[c]
            dw.writerow(descriptor)
            pic_num+=1
