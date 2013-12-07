#! /usr/bin/env python
# coding=utf-8

import picture_iterator as pit
import csv
import itertools
import math
from os import listdir
from os.path import isfile, join
import random


def split_merged_data(path="Data/dataset/merged_data.csv"):
    random.seed()
    with open("merged_data_train.csv", 'w') as train_file:
        with open("merged_data_test.csv", 'w') as test_file:
            with open('merged_data_val.csv', 'w') as val_file:
                with open(path, 'r') as data:
                    for line in data:
                        a = random.random()
                        if a < 0.6:
                            train_file.write(line)
                        elif a < 0.8:
                            val_file.write(line)
                        else:
                            test_file.write(line)


def distance(dot1,dot2):
    return math.sqrt((float(dot1[0]) - float(dot2[0]))**2
              + (float(dot1[1]) - float(dot2[1]))**2)


def group_points(keypoint_data, keypoint_names):
    points = [(keypoint_data['left_eye_outer_corner_x'], keypoint_data['left_eye_outer_corner_y']),
              (keypoint_data['right_eye_outer_corner_x'], keypoint_data['right_eye_outer_corner_y'])]
    for keyp in keypoint_names:
        if 'eye_outer_corner' in keyp[0]:
            continue
        points.append((keypoint_data[keyp[0]], keypoint_data[keyp[1]]))
    return points


def count_distances(points):
    distances = []
    norm = distance(points[0], points[1])
    for dot_idx, dot1 in enumerate(points[:-1]):
        for dot2 in points[dot_idx+1:]:
            distances.append((distance(dot1,dot2)/norm))
    return distances

def make_line(distances, emotion, number):
    distances = [str(i) for i in distances]
    result_line = [str(number), str(emotion)]
    result_line.extend(distances)
    result_line = ', '.join(result_line)
    return result_line

def most_common(L):
  groups = itertools.groupby(sorted(L))
  def _auxfun((item, iterable)):
    return len(list(iterable)), -L.index(item)
  return max(groups, key=_auxfun)[0]

def csv_merger(path_to_results = "../"):

    result_files = [ f for f in listdir(path_to_results) if isfile(join(path_to_results,f)) and f.find("result.csv") > -1 ]
    #merging emotions
    with open(join(path_to_results,"merged.csv"),'w') as merged_results_file:
        merged_results = csv.DictWriter(merged_results_file, ('number', 'emotion' ))
        for result_file_path in result_files:
            print ("merging file: "+result_file_path)
            with open(join(path_to_results,result_file_path),'r') as result_file:
                result = csv.DictReader(result_file, ('emotion','number'))
                for img in result:
                    merged_results.writerow(img)

    #sorting by pic number
    with open(join(path_to_results,"merged.csv"),'r') as merged_results_file:
        sorted_file = sorted(merged_results_file,key=lambda x:(int(x.split(',')[0])) )

    #saving to file with removing duplicates
    with open(join(path_to_results,"merged.csv"),'w') as merged_results_file:
        prev_pic_num = ''
        prev_emotions = []
        for line in sorted_file:
            words = line.split(',')
            if words[0] == prev_pic_num:
                prev_emotions.append(words[1])
            else:
                if prev_pic_num != '':
                    merged_results_file.write(','.join([prev_pic_num,most_common(prev_emotions)]) )
                prev_pic_num = words[0]
                prev_emotions = [words[1]]
        #last line
        merged_results_file.write(','.join([prev_pic_num,most_common(prev_emotions)]) )

    with open(join(path_to_results,'merged_data.csv'),"w") as merged_data:
        with open(join(path_to_results,"merged.csv"),'r') as merged_results_file:
            emotions_iterator = csv.DictReader(merged_results_file, ('number', 'emotion' ))

            pic_iterator = pit.PictureCollection(path=join(path_to_results,'training.csv'))
            dots = pic_iterator.key_points
            try:
                emotion = emotions_iterator.next()
                pic = pic_iterator.next()
                while True:
                    if int(emotion['number']) > int(pic['number']):
                        pic = pic_iterator.next()
                    elif int(emotion['number']) < int(pic['number']):
                        emotion = emotions_iterator.next()
                    else:
                        print ("Currently processing picture number: "+ str(pic["number"]))
                        grouped_p = group_points(pic, dots)
                        result_line = make_line(count_distances(grouped_p), emotion['number'], emotion['emotion'])

                        merged_data.write(result_line+'\n')

                        pic = pic_iterator.next()
                        emotion = emotions_iterator.next()
            except StopIteration:
                pass
    split_merged_data(join(path_to_results,'merged_data.csv'))



if __name__ == "__main__":
    csv_merger()
