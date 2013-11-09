__author__ = 'mark'
# coding=utf-8
import cv2
import cv
import numpy as np
CAMERA_INDEX = 0

#загружаем обученный датасеты
cascade = cv2.CascadeClassifier("Data/haarcascade_frontalface_default.xml")
cascade_leyes = cv2.CascadeClassifier("Data/haarcascade_mcs_lefteye.xml")
cascade_reyes = cv2.CascadeClassifier("Data/haarcascade_mcs_righteye.xml")
cascade_mouth = cv2.CascadeClassifier("Data/Mouth.xml")
cascade_nose = cv2.CascadeClassifier("Data/haarcascade_mcs_nose.xml")


#ищем лица
def detect_faces(image):
    faces = []
    detected = cascade.detectMultiScale(image, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))
    if len(detected) != 0:
        for (x, y, w, h) in detected:
            faces.append((x, y, w, h))
    return faces

#ищем глаза
def detect_eye(image):
    Eyes = []
    #ищем левые глаза
    detectedl = cascade_leyes.detectMultiScale(image, 1.3, 30, cv2.cv.CV_HAAR_SCALE_IMAGE, (10,10))
    if len(detectedl) != 0:
        for (x,y,w,h) in detectedl:
            Eyes.append((x, y, w, h))
    #ищем правые глаза
    detectedr = cascade_reyes.detectMultiScale(image, 1.3, 30, cv2.cv.CV_HAAR_SCALE_IMAGE, (10,10))
    #выбираем нехватающее
    if len(detectedl) != 0:
        for (x, y, w, h) in detectedr:
            f = 0
            for (x1,y1,w1,h1) in detectedl:
                if rectangle_intesect((x, y, w, h), (x1, y1, w1, h1)):
                    f = 1
                    break
            if f == 0:
                Eyes.append((x,y,w,h))


    return Eyes

#функция для простенькой проверки на пересечение прямоугольников.
# У нас не бывает, чтобы один лежал в другом, потому не проверяем
def rectangle_intesect((x1,y1,w1,h1),(x2,y2,w2,h2)):
    if x2>x1+w1 :
        return False
    elif x1 > x2+w2:
        return False
    elif y1 > y2+h2:
        return False
    elif y2 > y1+w1:
        return False
    return True
#ищем рты
def detect_mouth(image):
    mouth = []
    detected = cascade_mouth.detectMultiScale(image, 1.3, 40, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))
    if len(detected) != 0:
        for (x,y,w,h) in detected:
            mouth.append((x,y,w,h))
    return mouth
#ищем носы
def detect_nose(image):
    nose = []
    detected = cascade_nose.detectMultiScale(image, 1.3, 15, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))
    if len(detected) != 0:
        for (x,y,w,h) in detected:
            nose.append((x,y,w,h))
    return nose

#ищем углы рта
def CannyThreshold(lowThreshold,img):
    #Применяем Canny
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    detected_edges = cv2.GaussianBlur(gray,(3,3),0)
    detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
    x1 = 0
    x2 = img.shape[1]
    y1 = img.shape[0]/2
    y2 = y1
    #есил что-то нашлось - ищем дальше, иначе просто середины прямоугольника
    if dst is not None:
        imgray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    else:
        return (x1, y1), (x2, y2)
    ret,thresh = cv2.threshold(imgray,0,100,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dst,contours,-1,(0,255,0),2)
    dst = cv2.medianBlur(dst,5);
    #ищем ключевые точки
    for x in range(0,imgray.shape[1]):
        tmp = 0
        for y in range(0,imgray.shape[0]):
            if np.array_equal(dst[y,x],(0,255,0)):
                x1 = x
                y1 = y
                tmp = 1
                break
        if tmp:
            break
    for x in reversed(range(0,imgray.shape[1])):
        tmp = 0
        for y in reversed(range(0,imgray.shape[0])):
            if np.array_equal(dst[y,x],(0,255,0)):
                x2 = x
                y2 = y
                tmp = 1
                break
        if tmp:
            break
    cv2.imshow('canny demo',dst)
    return (x1, y1), (x2, y2)
if __name__ == "__main__":
    # получаем изображение
    cam = cv2.VideoCapture(0)
    storage = cv.CreateMemStorage()

    faces = []

    i = 0
    ratio = 3.5
    kernel_size = 3
    cv2.namedWindow('canny demo')
    while True:
        ret, image = cam.read()
        img = image.copy();

        #ищем раз в 10 кадров
        if i%10==0:
            faces = detect_faces(image)
            Eyes = detect_eye(image)
            mouth = detect_mouth(image)
            nose = detect_nose(image)
        #отрисовываем все что нашли
        for (x,y,w,h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), 255,2)


        for (x,y,w,h) in Eyes:
            cv2.rectangle(image, (x, y), (x+w, y+h),  (255, 255, 0))
            #img = image.copy();

            cv2.line(image, (x-2, (2*y+h)/2), (x+2, (2*y+h)/2), (0, 0, 0),2)
            cv2.line(image, (x, (2*y+h)/2 -2), (x, (2*y+h)/2 +2), (0, 0, 0),2)
            cv2.line(image, ((x+w)-2, (2*y+h)/2), ((x+w)+2, (2*y+h)/2), (0, 0, 0),2)
            cv2.line(image, ((x+w), (2*y+h)/2 -2), ((x+w), (2*y+h)/2 +2), (0, 0, 0),2)
            cv2.line(image, ((2*x+w)/2-2, (2*y+h)/2), ((2*x+w)/2+2, (2*y+h)/2), (0, 0, 0),2)
            cv2.line(image, ((2*x+w)/2, (2*y+h)/2 -2), ((2*x+w)/2, (2*y+h)/2 +2), (0, 0, 0),2)

        for (x,y,w,h) in mouth:
            f = 1
            for (X,Y,W,H) in Eyes:
                if rectangle_intesect((X,Y,W,H),(x,y,w,h)):
                    f = 0
            if f == 0:
                continue
            cv2.rectangle(image, (x, y), (x+w, y+h),  (255, 255, 255), 2)
            img = img[y:y+h, x:x+w]
            if i%10 == 0:
                (x1, y1), (x2, y2)= CannyThreshold(25,img)  # initialization
            cv2.line(image, (x+x1,y+ y1 -2), (x+x1,y+ y1 +2), (0, 0, 0),2)
            cv2.line(image, (x+x1 -2,y+ y1), (x+x1+2,y+ y1), (0, 0, 0),2)
            cv2.line(image, (x+x2,y+ y2 -2), (x+x2, y+y2 +2), (0, 0, 0),2)
            cv2.line(image, (x+x2 -2,y+ y2), (x+x2+2,y+ y2), (0, 0, 0),2)
            break
        for (x,y,w,h) in nose:
            cv2.rectangle(image, (x, y), (x+w, y+h),  (0, 255, 255), 2)
            cv2.line(image, ((2*x+w)/2-2, (2*y+h)/2), ((2*x+w)/2+2, (2*y+h)/2), (0, 0, 0),2)
            cv2.line(image, ((2*x+w)/2, (2*y+h)/2 -2), ((2*x+w)/2, (2*y+h)/2 +2), (0, 0, 0),2)
        cv2.namedWindow("gray", 1)
        cv2.imshow("gray", image)
        i += 1



        if(cv2.waitKey(10) == 27):
            break

