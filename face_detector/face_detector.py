#!/usr/bin/python
# -*- coding: utf-8 -*-

import os

import cv2

from DERP import config
from DERP.face_detector import helpers


class FaceDetector:
    def __init__(self):
        filename = 'haarcascade_frontalface_default.xml'
        path = os.path.join(config.CASCADES_DIR, filename)
        self.classifier = cv2.CascadeClassifier(path)
        self.webcam = None

    def debug_iterator(self):
        return self.__face_cam_stream(debug=True)

    def bounding_box_iterator(self):
        return self.__face_cam_stream()

    def image_iterator(self):
        return self.__face_cam_stream(return_images=True)

    def find_faces(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.medianBlur(gray_image, 5)
        width, height = gray_image.shape[1], gray_image.shape[0]

        desired_width = 300.
        scale = width/desired_width

        width, height = gray_image.shape[1], gray_image.shape[0]
        size = (int(width/scale), int(height/scale))
        scaled_image = cv2.resize(gray_image, size)

        found_faces = self.classifier.detectMultiScale(scaled_image, flags=cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT)
        scaled_faces = [[int(v*scale) for v in face] for face in found_faces]

        return scaled_faces

    @staticmethod
    def __debug(image, face_rects):
        for face_rect in face_rects:
            image = helpers.draw_bounding_boxes(image, face_rect)
        cv2.imshow('face capture', image)

        pressed = cv2.waitKey(10)
        if pressed == config.KEY_CODE_ESC:
            exit(0)

    def __face_cam_stream(self, debug=False, return_images=False):
        if self.webcam is None:
            self.webcam = helpers.configure_webcam()

        if not self.webcam.isOpened():
            raise StopIteration('Web cam is not opened')

        retval, image = self.webcam.read()
        while retval:
            image = cv2.flip(image, 1)
            face_rects = self.find_faces(image)

            if return_images:
                yield (image, helpers.cut_faces_from_images(image, face_rects))
            else:
                yield (image, face_rects)

            if debug:
                self.__debug(image, face_rects)

            retval, image = self.webcam.read()


if __name__ == '__main__':
    fd = FaceDetector()
    for (image, faces) in fd.debug_iterator():
        pass