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
    with open('result.csv','w') as result_file:
        dw = csv.DictWriter(result_file, ('emotion', 'number'))
        pic_num = 0;
        emotions = {'страх': 0, 'радость': 1, 'грусть': '3'}
        for pic in picture_iterator():
            cv2.imshow('e2', pic['Image'])
            descriptor = {'number': pic_num}
            descriptor['emotion'] = emotions['страх']
            dw.writerow(descriptor)

            c = cv2.waitKey(-1)
            if c== 1048603:
                break
            pic_num+=1
            if pic_num == 4:
                break
