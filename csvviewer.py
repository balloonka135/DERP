import csv
import cv2
import numpy as np

def main():
    with open('../training.csv', 'rb') as pictures_file:
        pictures = csv.DictReader(pictures_file)
        for picture in pictures:
            picture = [int(x) for x in picture['Image'].split(' ')]
            #print len(picture)
            picture = np.uint8(np.array(picture).reshape((96,96)))
            cv2.imshow('e2', picture)
            c = cv2.waitKey(1000)
            if c== 1048603:
                break


    pass

if __name__ == "__main__":
    main()