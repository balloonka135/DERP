#! /usr/bin/env python
# coding=utf-8

import picture_iterator as p_iter
import cv2
import numpy as np


def find_similar(path_to_data="Data/dataset/merged_data.csv"):
    p_collection = p_iter.PictureCollection()
    with open(path_to_data, 'r') as data_file:
        data_lines = data_file.readlines()
        for data_idx, data_line in enumerate(data_lines[:-1]):
            data_line = data_line.split(',')
            dot1 = np.array([float(x) for x in data_line[2:]])
            print ("processing image number: "+str(data_idx))
            for data2_line in data_lines[data_idx+1:]:
                data2_line = data2_line.split(',')
                dot2 = np.array([float(x) for x in data2_line[2:]])
                dist = np.linalg.norm(dot1-dot2)
                if dist < 0.15:
                    print (dist)
                    copy = np.zeros((96, 192), dtype=np.uint8)
                    copy[:, :96] = p_collection[int(data_line[0])]['Image']
                    copy[:, 96:] = p_collection[int(data2_line[0])]['Image']
                    cv2.imshow("similar", copy)
                    cv2.waitKey(-1)

if __name__ == '__main__':
    find_similar()
