#!/usr/bin/python
#coding=utf-8

#auhtor = sketchturner
#dataset reader

class DataReader(object):
    def __init__(self, path):
        self.path = path

    def read_generator(self):
        with open(self.path, 'rb') as fd:
            for data in fd:
                data = data.split(", ")
                yield data

    def get_classes(self):
        train_classes = []
        for data in self.read_generator():
            train_classes.append(int(data[1]))
        return train_classes

    def get_objects(self):
        train_objects = []
        temp = []
        for data in self.read_generator():
            temp=[float(i) for i in data[2:107]]
            train_objects.append(temp)
            temp = list()
        return train_objects

