#!/usr/bin/python
# -*- coding: utf-8 -*-

import face_detector
import cv2
import time
import numpy as np

# мега хак для получения текущего времени
current_milli_time = lambda: round(time.time() * 1000)


class FaceTracker:
    def __init__(self):
        self.detector = face_detector.FaceDetector()
        camera_index = 0
        self.capture = cv2.VideoCapture(camera_index)
        #всякие штюки для meanShift
        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        self.iters = 0;
        self._max_iters = 100;


    def fps(self, fn):
        self.frame_time = current_milli_time()
        old_frame_time = self.frame_time
        image = fn()
        frame_time = current_milli_time()
        fps = round(1000/(frame_time - old_frame_time+1))
        font = cv2.FONT_HERSHEY_COMPLEX #InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8) #Creates a font
        cv2.putText(image, str(fps), (50,50),font, 1.0,(200,200,200))
        return image

    def faces(self):
        _,frame = self.capture.read()
        if (self.iters == 0):
            faces = self.detector.find_faces(frame)
            if len(faces)<1:
                return frame
            self.track_window = tuple(faces[0])
            self.roi = frame[faces[0][0]:faces[0][0]+faces[0][2], faces[0][1]:faces[0][1]+faces[0][3]]
            if (len(faces)<1):
                frame = self.detector.draw_faces(frame,faces)
                return frame
        self.iters +=1
        if self.iters == self._max_iters:
            self.iters = 0

        hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)

        # Draw it on image
        x,y,w,h = self.track_window
        #cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        """try:
            if (len(self.detector.find_faces(frame[x+10:x+w+10][y+10:y+h+10]))):
                self.iters = 0
            else:
                cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        except Exception:
            self.iters = 0"""
        return frame

        """
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask = np.uint8(np.zeros(gray.shape))
        for face in faces:
            cv2.rectangle(mask,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,),-1)
        if len(faces) > 0:
            pass
        """
        """
            features = cv2.goodFeaturesToTrack(gray,1000,0.001,10,mask = mask)
            ret,labels, centers = cv2.kmeans(features,len(faces),None,(cv2.TERM_CRITERIA_COUNT,100,1),100,cv2.KMEANS_RANDOM_CENTERS)
            for feat in features:
                cv2.circle(frame,tuple(feat[0]),10,(255,0,0))
            for cent in centers:
                cv2.circle(frame,tuple(cent),20,(0,0,255),-1)
        """


if __name__ == '__main__':
    ft = FaceTracker()
    while (True):
        cv2.imshow('e2',ft.fps(ft.faces))
        c= cv2.waitKey(10)
        if c > 0:
            if (c == 1048603):
                break

