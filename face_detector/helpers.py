import cv2


def configure_webcam(width=640, height=480):
    webcam = cv2.VideoCapture(0)

    webcam.set(cv2.cv.CV_CAP_PROP_CONVERT_RGB, False)
    webcam.set(cv2.cv.CV_CAP_PROP_FOURCC, False)
    webcam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
    webcam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)

    return webcam


def cut_faces_from_images(image, face_descriptions):
    images = []
    for face in face_descriptions:
        x, y, w, h = face
        images.append(image[y:y+h, x:x+w])
    return images


def draw_bounding_boxes(image, face_descriptions):
    color = (0, 0, 255)

    for face in face_descriptions:
        x, y, w, h = face

        top_left = (x, y)
        bottom_right = (x+w, y+h)

        cv2.rectangle(image, top_left, bottom_right, color)

    return image