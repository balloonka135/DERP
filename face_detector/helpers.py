import cv2


def configure_webcam(width=640, height=480):
    webcam = cv2.VideoCapture(0)

    webcam.set(cv2.cv.CV_CAP_PROP_CONVERT_RGB, False)
    webcam.set(cv2.cv.CV_CAP_PROP_FOURCC, False)
    webcam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
    webcam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)

    return webcam


def cut_face_from_images(image, face_rect):
    x, y, w, h = face_rect
    return image[y:y+h, x:x+w]


def cut_faces_from_images(image, face_rects):
    return [cut_face_from_images(image, face_rect) for face_rect in face_rects]


def draw_bounding_boxes(image, face_rect):
    color = (0, 0, 255)

    x, y, w, h = face_rect

    top_left = (x, y)
    bottom_right = (x+w, y+h)

    cv2.rectangle(image, top_left, bottom_right, color)

    return image