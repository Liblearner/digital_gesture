import os

import cv2
import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import *


def get_file(file_dir):
    one = []
    label_one = []
    two = []
    label_two = []
    three = []
    label_three = []
    four = []
    label_four = []
    five = []
    label_five = []
    ok = []
    label_six = []
    six = []
    label_seven = []
    seven = []
    label_eight = []
    eight = []
    label_nine = []
    nine = []
    label_ten = []
    ten = []

    label_ok = []

    good = []
    label_good = []

    for file in os.listdir(file_dir + '/ONE'):
        one.append(file_dir + '/ONE' + '/' + file)
        label_one.append(0)
    for file in os.listdir(file_dir + '/TWO'):
        two.append(file_dir + '/TWO' + '/' + file)
        label_two.append(1)
    for file in os.listdir(file_dir + '/THREE'):
        three.append(file_dir + '/THREE' + '/' + file)
        label_three.append(2)
    for file in os.listdir(file_dir + '/FOUR'):
        four.append(file_dir + '/FOUR' + '/' + file)
        label_four.append(3)
    for file in os.listdir(file_dir + '/FIVE'):
        five.append(file_dir + '/FIVE' + '/' + file)
        label_five.append(4)
    for file in os.listdir(file_dir + '/SIX'):
        six.append(file_dir + '/SIX' + '/' + file)
        label_six.append(5)
    for file in os.listdir(file_dir + '/SEVEN'):
        seven.append(file_dir + '/SEVEN' + '/' + file)
        label_seven.append(6)
    for file in os.listdir(file_dir + '/EIGHT'):
        eight.append(file_dir + '/EIGHT' + '/' + file)
        label_eight.append(7)
    for file in os.listdir(file_dir + '/NINE'):
        nine.append(file_dir + '/NINE' + '/' + file)
        label_nine.append(8)
    for file in os.listdir(file_dir + '/TEN'):
        ten.append(file_dir + '/TEN' + '/' + file)
        label_ten.append(9)
    for file in os.listdir(file_dir + '/OK'):
        ok.append(file_dir + '/OK' + '/' + file)
        label_ok.append(10)
    for file in os.listdir(file_dir + '/GOOD'):
        good.append(file_dir + '/GOOD' + '/' + file)
        label_good.append(11)

    image_list = np.hstack((one, two, three, four, five, six, seven, eight, nine, ten, ok, good))
    label_list = np.hstack((label_one, label_two, label_three, label_four, label_five, label_six, label_seven,
                            label_eight, label_nine, label_ten, label_ok, label_good))
    temp = np.array([image_list, label_list])  # 转换成2维矩阵
    temp = temp.transpose()  # 转置

    np.random.shuffle(temp)  # 按行随机打乱顺序函数

    return image_list, label_list


def image_to_h5(X_dirs, Y):
    counter = 0
    X = []
    for dirs in X_dirs:
        counter = counter + 1
        im = cv2.imread(dirs, 0)
        print("正在处理第%d张照片" % counter)
        # resize_im = cv2.resize(im,(40,40),interpolation= cv2.INTER_AREA)
        # img_gray = cv2.cvtColor(resize_im,cv2.COLOR_RGB2GRAY)
        mat = np.asarray(im)  # image 转矩阵
        X.append(mat)

    aa = np.array(X)
    num, _, _ = aa.shape
    aa.reshape(num, 40, 40, 1)
    print(aa.shape)

    file = h5py.File("dataset//data_notwhite.h5", "w")
    file.create_dataset('X', data=aa)
    file.create_dataset('Y', data=np.array(Y))
    file.close()


# test
# data = h5py.File("dataset//data.h5","r")
# X_data = data['X']
# print(X_data.shape)
# Y_data = data['Y']
# print(Y_data[123])
# image = Image.fromarray(X_data[123]) #矩阵转图片并显示
# image.show()


if __name__ == "__main__":
    # print("start.....: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
    # resize_img()
    # print("end....: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
    train_dir = 'E:/Project/DeepLearning/anph1a//hand_gesture_dataset'
    train, train_label = get_file(train_dir)
    image_to_h5(train, train_label)
    # test

    data = h5py.File("dataset//data_notwhite.h5", "r")
    X_data = data['X']
    print(X_data.shape)
    Y_data = data['Y']
    print(Y_data[1235])
    # image = Image.fromarray(X_data[1235])  # 矩阵转图片并显示
    # image.show()
