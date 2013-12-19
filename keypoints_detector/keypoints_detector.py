#!/usr/bin/python

import os
import cv2

import pyasm
from DERP import config
from DERP.face_detector import face_detector
from DERP.face_detector import helpers


class KeypointsDetector(object):
    def __init__(self):
        model_file = 'muct76.model'
        path = os.path.join(config.DATA_DIR, model_file)
        self.model = pyasm.AsmModel(path)

    def find_one(self, image, face_rect):
        x, y, w, h = face_rect

        keypoints = self.model.fit_one(helpers.cut_face_from_images(image, face_rect))
        keypoints[::2] += x
        keypoints[1::2] += y

        return keypoints

    def find_all(self, image, face_rects):
        return [self.find_one(image, face_rect) for face_rect in face_rects]


if __name__ == '__main__':
    fd = face_detector.FaceDetector()
    kd = KeypointsDetector()
    for (image, faces) in fd.bounding_box_iterator():
        keypoints = kd.find_one(image, faces[0])
        for (x, y) in zip(keypoints[::2], keypoints[1::2]):
            cv2.circle(image, (x, y), 1, (255, 0, 0), 2)

        cv2.imshow('debug', image)
        pressed = cv2.waitKey(10)
        if pressed == config.KEY_CODE_ESC:
            exit(0)