# coding=utf-8

import os
import sys
import random
import fileinput


def gen_label():
    """"""
    # imagedata dir
    img_dir = "/home/nd/datasets/furniture/"

    train_f = open("train.txt", 'w')
    val_f = open("val.txt", 'w')

    dir_num = 0
    for sub_dir in os.listdir(img_dir):

        if os.path.isfile(sub_dir):
            continue

        dir_num += 1
        label = int(sub_dir) - 1
        img_path = os.path.join(img_dir, sub_dir)

        file_list = os.listdir(img_path)
        val_file_list = random.sample(file_list, 3)

        for file in file_list:
            file_path_label = img_path + "/" + file + ' ' + str(label)

            if file in val_file_list:
                val_f.write(file_path_label + '\n')
            else:
                train_f.write(file_path_label + '\n')

    return dir_num


def main():
    """修改prototxt文件中的类别数"""
    class_num = gen_label()

    args = sys.argv
    for i in range(2):
        file_name = args[i+1]
        with open(file_name, 'r') as rf:
            data = rf.readlines()

        wf = open(file_name, 'w')
        line_num = 10000
        for i, line in enumerate(data):
            cont_line = line.strip()

            if cont_line.startswith("inner_product_param"):
                line_num = i
                print(line_num)

            if i > line_num and cont_line.startswith("num_output"):
                new_cont_line = "num_output: {}".format(str(class_num))
                line = line.replace(cont_line, new_cont_line)

            wf.write(line)


if __name__ == '__main__':

    main_2()