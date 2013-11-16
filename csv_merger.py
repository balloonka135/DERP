#! /usr/bin/env python
# coding=utf-8

import csvviewer
import csv
import itertools
import math
from os import listdir
from os.path import isfile, join

def distance(dot1,dot2, dots_dict):
    return math.sqrt((float(dots_dict[dot1[0]]) - float(dots_dict[dot2[0]]))**2
              + (float(dots_dict[dot1[1]]) - float(dots_dict[dot2[1]]))**2)

def count_distances(dots_dict, dots):
    norm_dot1 = ('left_eye_outer_corner_x', 'left_eye_outer_corner_y')
    norm_dot2 = ('right_eye_outer_corner_x', 'right_eye_outer_corner_y')
    norm = distance(norm_dot1, norm_dot2, dots_dict)
    distances = []
    for dot_idx,dot1 in enumerate(dots[:-1]):
        for dot2 in dots[dot_idx+1:]:
            distances.append(str(distance(dot1,dot2,dots_dict)/norm))
    return distances


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
            dots = []
            pic_iterator = csvviewer.picture_dot_iterator(path=join(path_to_results,'training.csv'),dots = dots)
            try:
                emotion = emotions_iterator.next()
                pic = pic_iterator.next()
                while True:
                    if int(emotion['number']) > int(pic['number']):
                        pic = pic_iterator.next()
                    elif int(emotion['number']) < int(pic['number']):
                        emotion = emotions_iterator.next()
                    else:
                        print ("Currently processing picture number: "+ pic["number"])
                        distances  = count_distances(pic, dots)
                        #try:
                        line = [emotion['number'],emotion['emotion']]
                        line.extend(distances)

                        result_line = ', '.join(line)
                        merged_data.write(result_line+'\n')

                        pic = pic_iterator.next()
                        emotion = emotions_iterator.next()
            except StopIteration:
                pass



if __name__ == "__main__":
    csv_merger()