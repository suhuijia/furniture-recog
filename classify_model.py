# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import os
import argparse

# caffe_root = "/home/nd/caffe_master/"
# net_file = caffe_root + "examples/resnet/resnet_deploy.prototxt"
# model = caffe_root + "examples/resnet/model/resnet_iter_25000.caffemodel"
# caffe.set_mode_cpu()
# net = caffe.Net(net_file, model, caffe.TEST)
# image_mean = np.load(caffe_root + "examples/resnet/furniture_mean.npy").mean(1).mean(1)

def parse_args():
    parser = argparse.ArgumentParser(description='calssify model test')
    # general
    parser.add_argument('--caffe_root', default="/home/nd/caffe_master/", help='')
    parser.add_argument('--net_file', default='examples/resnet/resnet_deploy.prototxt', help='')
    parser.add_argument('--model_file', default='examples/resnet/model/resnet_v3_iter_64200.caffemodel', help='')
    parser.add_argument('--mean_file', default='examples/resnet/furniture_mean.npy', help='')
    args = parser.parse_args()
    return args

def load_model():
    args = parse_args()
    global net, image_mean
    net_file = args.caffe_root + args.net_file
    model_file = args.caffe_root + args.model_file
    mean_file = args.caffe_root + args.mean_file
    caffe.set_mode_cpu()
    net = caffe.Net(net_file, model_file, caffe.TEST)
    image_mean = np.load(mean_file).mean(1).mean(1)

    return net, image_mean


def furniture_classify(image_file_path, net, image_mean, top_n=10):

    # top_n = input("please input a number, denotes the previous categories: ")

    if isinstance(top_n, int):
        pass
    else:
        top_n = int(top_n)

    key = "top_{}".format(str(top_n))

    result = {}

    data_shape = net.blobs['data'].data.shape
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', image_mean)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    image = caffe.io.load_image(image_file_path)
    transformed_image = transformer.preprocess('data', image)

    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    output_prob = output['prob'][0]
    # print 'The predicted class is : ', output_prob.argmax()

    top_1 = output_prob.argsort()[::-1][0]
    # print(top_1)
    top_n = output_prob.argsort()[::-1][0:top_n]
    top_n = top_n.tolist()

    print(top_n)

    top_n_sim = []
    for ele in top_n:
        # print(ele)
        # print(ele, type(ele))
        ele_str = str(ele+1).zfill(5)
        # print(ele_str, type(ele_str))
        # top_n_sim[ele_str] = str(output_prob[ele])
        # sub_str = "{}: {}".format(ele_str, str(output_prob[ele]))
        # top_n_sim.append(sub_str)
        top_n_sim.append((ele_str, "%f"%output_prob[ele]))

    prob_1 = output_prob[top_1]
    # print 'The probabilities is : ', prob_1
    result["maxSimilarity"] = str(prob_1)

    result[key] = top_n_sim

    # print(result)
    return result


if __name__ == '__main__':

    image_file_path = "/home/nd/caffe_master/examples/resnet/test_image/60473.00_18.png"

    result = furniture_classify(image_file_path)

