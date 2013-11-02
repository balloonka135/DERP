#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2


class FaceDetector:
    ESC_CODE = 27

    def __init__(self, path='Data/lbpcascade_frontalface.xml'):
        self.classifier = cv2.CascadeClassifier(path)
        self.webcam = None

    @staticmethod
    def configure_webcam(width=640, height=480):
        webcam = cv2.VideoCapture(0)

        webcam.set(cv2.cv.CV_CAP_PROP_CONVERT_RGB, False)
        webcam.set(cv2.cv.CV_CAP_PROP_FOURCC, False)
        webcam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
        webcam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)

        return webcam

    @staticmethod
    def draw_faces(image, faces):
        color = (0, 0, 255)

        for face in faces:
            x, y, w, h = face

            top_left = (x, y)
            bottom_right = (x+w, y+h)

            cv2.rectangle(image, top_left, bottom_right, color)

        return image

    def find_faces(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.medianBlur(gray_image,5)
        width, height = gray_image.shape[1], gray_image.shape[0]

        desired_width = 300.
        scale = width/desired_width

        width, height = gray_image.shape[1], gray_image.shape[0]
        size = (int(width/scale), int(height/scale))
        scaled_image = cv2.resize(gray_image, size)

        faces = self.classifier.detectMultiScale(scaled_image)
        faces = [map(lambda v: int(v*scale), face) for face in faces]

        return faces

    @staticmethod
    def get_faces_images(image, faces):
        images = []
        for face in faces:
            x, y, w, h = face
            images.append(image[y:y+h, x:x+w])
        return images

    def face_cam_stream(self, debug=False, return_images=False):
        if self.webcam is None:
            self.webcam = FaceDetector.configure_webcam()

        if self.webcam.isOpened():
            retval, image = self.webcam.read()
        else:
            retval = False

        while retval:
            faces_description = self.find_faces(image)

            if return_images:
                yield FaceDetector.get_faces_images(image, faces_description)
            else:
                yield faces_description

            if debug:
                image = FaceDetector.draw_faces(image, faces_description)
                cv2.imshow('face capture', image)

            pressed = cv2.waitKey(10)
            if pressed == FaceDetector.ESC_CODE:
                exit(0)

            retval, image = self.webcam.read()

if __name__ == '__main__':
    fd = FaceDetector()
    for faces in fd.face_cam_stream(debug=True):
        pass