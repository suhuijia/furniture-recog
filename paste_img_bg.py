# coding=utf-8

import os
import sys
import cv2
import random
import time

orig_root = "/home/nd/caffe_master/examples/resnet/test_data_231_0.6/"

list_dir = os.listdir(orig_root)


def remove_img(orig_root):
    for sub_dir in os.listdir(orig_root):
        if sub_dir.startswith('00'):
            print(sub_dir)
            sub_dir_path = os.path.join(orig_root, sub_dir)
            img_list = random.sample(os.listdir(sub_dir_path), 2)

            for file in os.listdir(sub_dir_path):
                if file not in img_list:
                    remove_file = os.path.join(sub_dir_path, file)
                    os.remove(remove_file)
                else:
                    continue

        else:
            pass


def dir_choice(list_dir):

    while True:
        sub_dir = random.choice(list_dir)
        if sub_dir.startswith('00'):
            continue
        else:
            break

    return sub_dir


for sub_dir in list_dir:
    print(sub_dir)

    if sub_dir.startswith("00"):
        sub_dir_path = os.path.join(orig_root, sub_dir)
        for file in os.listdir(sub_dir_path):

            re_sub_dir = dir_choice(list_dir)
            re_sub_dir_path = os.path.join(orig_root, re_sub_dir)
            
            re_file = random.choice(os.listdir(re_sub_dir_path))
            re_file_path = os.path.join(re_sub_dir_path, re_file)
            re_img = cv2.imread(re_file_path)
            re_h, re_w, channel = re_img.shape

            
            file_path = os.path.join(sub_dir_path, file)
            img = cv2.imread(file_path)

            h, w, channel = img.shape
            print(re_h, h)
            if re_h > h:
                size = int(re_h*0.6)
                img_resize = cv2.resize(img, (size, size))
                print(img_resize.shape)
                pad = int((re_h - size)/2)
                print(pad)
                re_img[pad:pad+size, pad:pad+size] = img_resize

            else:
                size = int(re_h*0.6)
                img_resize = cv2.resize(img, (size, size))
                pad = int((re_h - size)/2)
                re_img[pad:pad+size, pad:pad+size] = img_resize

            cv2.imwrite(file_path, re_img)
            time.sleep(0.01)
            # cv2.imshow('paste', re_img)
            # cv2.waitKey(2000)
