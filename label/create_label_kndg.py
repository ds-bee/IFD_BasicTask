import numpy as np
import os, sys, time
from random import shuffle
from math import floor


label_path = r'D:\IFD_BasicTask\label\DONG_label'
label_name = '/label.txt'
class_name_list = ['chip1a', 'chip2a', 'chip3a', 'chip4a', 'chip5a', 'crack', 'healthy', 'missing', 'spall']
class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]


def total_label(path):
    """
    创建并保存所有样本的路径以及标签到各个类别下的 label.txt；
    """
    root, dirs, files = next(os.walk(path))
    for j in range(len(class_list)):
        print(' %s \t' % (dirs[j]), end='|\n')
    for i in range(len(dirs)):
        f = open(os.path.join(label_path, dirs[i] + '.txt'), 'w', encoding='utf-8')
        j = class_list[i]   # i类型
        fs = os.listdir(path + '/' + dirs[i])
        for file in fs:
            line = (path + '/' + str(dirs[i]) + '/' + str(file) + ' ' + str(j) + '\n')
            f.write(line)
        f.close()


def divide_label(path):
    """
    根据输入类的样本txt，返回两个列表，包含该类的训练与测试样本路径和标签；
    """
    train = []
    test = []
    label = []
    with open(path, 'r', encoding='utf-8') as f:
        row_list = f.readlines()
    for j in row_list:
        label.append(j)
    shuffle(label)
    length = len(label)
    train = label[:floor(length * 7 / 10)]
    test = label[floor(length * 7 / 10):]
    return train, test


def shuffle_label(tra_file, tes_file):
    with open(tra_file, 'r', encoding='utf-8') as ftra:
        label_tra = ftra.readlines()
        shuffle(label_tra)
    ftr = open(os.path.join(label_path, 'kndg_shuffle_train.txt'), 'a', encoding='utf-8')
    for i in label_tra:
        ftr.write(i)
    ftr.close()
    with open(tes_file, 'r', encoding='utf-8') as ftes:
        label_tes = ftes.readlines()
        shuffle(label_tes)
    fte = open(os.path.join(label_path, 'kndg_shuffle_test.txt'), 'a', encoding='utf-8')
    for j in label_tes:
        fte.write(j)
    fte.close()


def remake_label(path):
    label = []
    with open(path + label_name, 'r', encoding='utf-8') as f:
        for i in f.readlines():
            label.append(i)
    shuffle(label)      # 打乱
    length = len(label)
    print(length, end='\n')       # 检查长度200 * 4
    # 分配数据
    train_label = label[:floor(length * 5 / 10)]
    test_label = label[floor(length * 5 / 10):]
    ftr = open(path + label_path + '/train_label_' + '.txt', 'w', encoding='utf-8')
    fte = open(path + label_path + '/test_label_' + '.txt', 'w', encoding='utf-8')
    for line in train_label:
        ftr.write(line)
    for line in test_label:
        fte.write(line)


def mixed_label(label_type, label_number):
    label_list = []
    exd = '.txt'
    with open('./gear_diagnose/gear_labels/' + label_type + '_label_' + str(label_number) +
        exd, 'r', encoding='utf-8' ) as f:
        for i in f.readlines():
            label_list.append(i)
        # if not len(label_list) == 500:
        if not len(label_list) == 400:
            print('There must be sth wrong!')
        f.close()
    # 将列表添加到文件
    with open('./gear_diagnose/gear_labels/mixed_' + label_type + exd, 'a', encoding='utf-8') as g:
        for i in label_list:
            g.write(i)
        g.close()


def main():
    path = 'E:/AAA/Projects/IFD_BasicTask/dataset/KNDG'

    # total_label(path)
    # time.sleep(0.1)

    train = []
    test = []
    for j in class_name_list:
        filename = j + '.txt'
        tra, tes = divide_label(os.path.join(label_path, filename))
        # train.append(i for i in tra)
        train.extend(tra)
        test.extend(tes)
    ftr = open(label_path + '/' + 'kndg_train.txt', 'a', encoding='utf-8')
    fte = open(label_path + '/' + 'kndg_test.txt', 'a', encoding='utf-8')
    for line in train:
        ftr.write(line)
    ftr.close()         # 来不及完成文件操作，就进行下一步操作，解决办法：添加文件关闭操作
    for line in test:
        fte.write(line)
    fte.close()

    filename_tra = label_path + '/' + 'kndg_train.txt'
    filename_tes = label_path + '/' + 'kndg_test.txt'
    shuffle_label(filename_tra, filename_tes)
    print('Make finished')


if __name__ == '__main__':
    main()

