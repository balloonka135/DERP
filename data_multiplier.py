#coding=utf-8

import picture_iterator
import csv_merger
from math import sin, cos, radians

CENTER = (45, 45)

def rotate(point, angle):
    x = float(point[0]) - CENTER[0]
    y = float(point[1]) - CENTER[1]
    nx = (x*cos(radians(angle))) - (y*sin(radians(angle)))
    ny = (x*sin(radians(angle))) + (y*cos(radians(angle)))
    new_point = (nx + CENTER[0], ny + CENTER[1])
    return new_point

def scale(point, dx, dy):
    x = float(point[0]) - CENTER[0]
    y = float(point[1]) - CENTER[1]
    nx = x*dx
    ny = y*dy
    new_point = (nx + CENTER[0], ny + CENTER[1])
    return new_point



def make_lots_of_data(path_to_pictures, path_to_marks = "Data/dataset/merged.csv", path_to_new_data = "Data/dataset/moar.csv"):
    pic_it = picture_iterator.PictureCollection()
    points_desc = pic_it.key_points
    counter = 0
    angles = [3,6,9,12,15,-3,-6,-9,-12,-15]
    dxs = [1.02,1.04,1.06,1.10]
    dys = [1.02,1.04,1.06,1.10]
    with open(path_to_new_data,'w') as new_data:
        with open(path_to_marks) as marks:
            for line in marks:
                line = line.split(',')
                pic_num = int(line[0])
                emo_num = int(line[1])
                pic = pic_it[pic_num]
                if '' in pic.values():
                    continue
                counter += 1
                print ("image num %s."%counter)

                grouped_p = csv_merger.group_points(pic, points_desc)
                new_data.write(csv_merger.make_line(csv_merger.count_distances(grouped_p), emotion= emo_num, number=pic_num)+'\n')
                for angle in angles:
                    rotated_p = [rotate(i,angle) for i in grouped_p]
                    new_data.write(csv_merger.make_line(csv_merger.count_distances(rotated_p), emotion= emo_num, number=pic_num)+'\n')
                for dx in dxs:
                    for dy in dys:
                        scaled_p = [scale(i, dx, dy) for i in grouped_p]
                        new_data.write(csv_merger.make_line(csv_merger.count_distances(scaled_p), emotion= emo_num, number=pic_num)+'\n')



if __name__ == "__main__":
    make_lots_of_data(path_to_pictures=None)