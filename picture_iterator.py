#! /usr/bin/env python
# coding=utf-8

import csv
import cv2
import numpy as np

import config


class PictureCollection:
    def __init__(self, path='../training.csv', start=0, end=-1, filter_partial_descs = True):
        self.collection = []
        self.pictures_file = open(path, 'rb')
        self.pictures_reader = csv.DictReader(self.pictures_file)
        self.image_iterator_pointer = start
        self.max_image_iterator_pointer = end

        all_keys = list(self.pictures_reader.fieldnames)
        del all_keys[all_keys.index('Image')]
        self.key_points = [(dot1, dot2) for dot1 in all_keys for dot2 in all_keys
                           if dot2[:-1] == dot1[:-1] and dot1[-1] == 'x' and dot2[-1] == 'y']
        self.filter_partial_descs = filter_partial_descs

    def __del__(self):
        self.pictures_file.close()

    def __getitem__(self, item):
        while item >= len(self.collection):
            try:
                picture = self.pictures_reader.next()
                image = [int(x) for x in picture['Image'].split(' ')]
                picture['Image'] = np.uint8(np.array(image).reshape((96, 96)))
                self.collection.append(picture)
            except StopIteration:
                raise IndexError
        return self.collection[item]

    def __iter__(self):
        return self

    def next(self):
        while True:
            if self.image_iterator_pointer >= self.max_image_iterator_pointer > 0:
                raise StopIteration
            try:
                picture = self.__getitem__(self.image_iterator_pointer)
            except IndexError:
                raise StopIteration

            if self.filter_partial_descs and '' in picture.values():
                continue

            picture['number'] = self.image_iterator_pointer
            self.image_iterator_pointer += 1
            return picture


