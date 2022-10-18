import numpy as np
import os, sys, time
from random import shuffle
from math import floor

PATH = r'/dataset/CWRU'
S_PATH = r'/label/CWRU_label'


def l_total(path, dir_l, num):
    """
    create txt label
    """
    txt_list = []
    class_list = ["B007", "B014", "B021", "IR007", "IR014", "IR021", "OR0076", "OR0146", "OR0216", "normal"]
    cls_dir = os.path.join(path, dir_l)
    # root, dirs, files = next(os.walk(cls_dir))
    for i in range(len(class_list)):
        file_list = os.listdir(os.path.join(cls_dir, class_list[i]+str(num)))
        with open(os.path.join(S_PATH, dir_l+"_"+class_list[i]+str(num)+'.txt'), 'a', encoding='utf-8') as f:
            for j in file_list:
                line = os.path.join(cls_dir, class_list[i]+str(num), j) + ' ' + str(i) + '\n'
                f.write(line)
        txt_list.append(os.path.join(S_PATH, dir_l+"_"+class_list[i]+str(num)+'.txt'))
    return txt_list


def l_divide(path, pngx):
    label = []
    train = []
    test = []
    with open(path, 'r', encoding='utf-8') as file:
        for i in file.readlines():
            label.append(i)
    # print(len(label))
    train = label[:floor(len(label)*0.7)]
    test = label[floor(len(label)*0.7):]
    with open(os.path.join(S_PATH, 'png_' + str(pngx) + '_cwru_train.txt'), 'a', encoding='utf-8') as ftr:
        for i in train:
            ftr.write(i)
    with open(os.path.join(S_PATH, 'png_' + str(pngx) + '_cwru_test.txt'), 'a', encoding='utf-8') as fte:
        for j in test:
            fte.write(j)


def shuffle_label(path=S_PATH, cls_name=""):
    label = []
    with open(os.path.join(path, cls_name+'.txt'), 'r', encoding='utf-8') as f:
        for i in f.readlines():
            label.append(i)
    shuffle(label)
    with open(os.path.join(path, cls_name+'.txt'), 'w', encoding='utf-8') as f:
        for i in label:
            f.write(i)


if __name__ == '__main__':
    if not os.path.exists(S_PATH):
        os.makedirs(S_PATH)
    for i in range(4):
        txt = l_total(path=PATH, dir_l='png_'+str(i), num=i)
        for j in txt:
            l_divide(j, i)
        shuffle_label(cls_name='png_' + str(i) + '_cwru_train')
        shuffle_label(cls_name='png_' + str(i) + '_cwru_test')
