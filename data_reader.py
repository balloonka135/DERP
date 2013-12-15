#!/usr/bin/python
#coding=utf-8

class DataReader(object):
    def __init__(self, path):
        self.path = path

    def read_generator(self):
        with open(self.path, 'rb') as fd:
            for data in fd:
                data = data.split(", ")
                yield data

    def get_classes(self):
        if hasattr(self, "train_classes"):
            return self.train_classes
        else:
            self.train_classes = []
            for data in self.read_generator():
                self.train_classes.append(int(data[1]))
            return self.train_classes

    def get_objects(self):
        if hasattr(self, "train_objects"):
            return self.train_objects
        self.train_objects = []
        temp = []
        for data in self.read_generator():
            temp=[float(i) for i in data[2:107]]
            self.train_objects.append(temp)
            temp = list()
        return self.train_objects

