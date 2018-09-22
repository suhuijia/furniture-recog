# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import pylab
from PIL import Image
import caffe
import os


def proto_to_npy():
    MEAN_PROTO_PATH = '/home/nd/shj/project/RA_CNN_caffe/examples/imagenet/flower_mean.binaryproto' 
    MEAN_NPY_PATH = '/home/nd/shj/project/RA_CNN_caffe/examples/imagenet/flower_mean.npy'           

    blob = caffe.proto.caffe_pb2.BlobProto()           # 创建protobuf blob
    data = open(MEAN_PROTO_PATH, 'rb' ).read()         # 读入mean.binaryproto文件内容
    blob.ParseFromString(data)                         # 解析文件内容到blob

    array = np.array(caffe.io.blobproto_to_array(blob))# 将blob中的均值转换成numpy格式，array的shape （mean_number，channel, hight, width）
    mean_npy = array[0]                                # 一个array中可以有多组均值存在，故需要通过下标选择其中一组均值
    np.save(MEAN_NPY_PATH ,mean_npy)


def main():
    test_file_path = "/home/nd/datasets/FGVC_Flower/train/test_0.txt"
    with open(test_file_path, 'r') as rf:
        data = rf.readlines()

    image_dir = "/home/nd/shj/project/RA_CNN_caffe/examples/googlenet/img_224/"
    caffe_root = "/home/nd/shj/project/RA_CNN_caffe/"
    net_file = caffe_root + "examples/googlenet/deploy_test.prototxt"
    model = caffe_root + "examples/googlenet/model/flower_googlenet_iter_752000.caffemodel"

    caffe.set_mode_gpu()
    net = caffe.Net(net_file, model, caffe.TEST)
    image_mean = np.load(caffe_root + "examples/imagenet/flower_mean.npy").mean(1).mean(1)

    # print 'mean-subtracted values:', zip('RGB', image_mean)
    # 输出结果：mean-subtracted values: [('R', 104.0069879317889), ('G', 116.66876761696767), ('B', 122.6789143406786)]

    data_shape = net.blobs['data'].data.shape
    # print data_shape
    # 输出结果：(10, 3, 224, 224) batch_size:10  channels:3  height:224  weight:224

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', image_mean)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    i = 0
    # result_f = open("result_val_1.txt", 'w')

    for line in data:
        line = line.strip()
        file = line.split(' ')[0]
        label = line.split(' ')[1]
        file_path = os.path.join(image_dir, file)
        file_name = file.split('.')[0].split('/')[1]
        # file_path = raw_input("file path: ")
        image = caffe.io.load_image(file_path)
        transformed_image = transformer.preprocess('data', image)
        # plt.imshow(image)
        # pylab.show()

        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        output_prob = output['prob'][0]
        # print(output_prob.shape)
        print 'The predicted class is : ', output_prob.argmax()
        # print(output_prob.argsort())
        top_1 = output_prob.argsort()[::-1][0]
        # print(top_1)
        prob_1 = output_prob[top_1]
        print 'The probabilities is : ', prob_1
        # result_str = file_name + " " + label + " " +  str(top_1) + " " + str(prob_1)
        # result_f.write(result_str + '\n')

        if label == str(top_1):
            i += 1
    # result_f.close()

    print(i)
    # # 输出对应的类别以及top_n的结果
    # label_file = caffe_root + "data/ilsvrc12/synset_words.txt"
    # labels = np.loadtxt(label_file, str, delimiter='\t')
    # print 'The label is : ', labels[output_prob.argmax()]

    # top_inds = output_prob.argsort()[::-1][:5]
    # print 'probabilities and labels: ', zip(output_prob[top_inds], labels[top_inds])



def main_test():

    image_dir = "/home/nd/caffe_master/examples/resnet/jiaju_test/"

    # googlenet 网络模型
    # caffe_root = "/home/nd/caffe_master/"
    # net_file = caffe_root + "examples/googlenet/deploy_test.prototxt"
    # model = caffe_root + "examples/googlenet/model/furniture_googlenet_iter_30000.caffemodel"


    # resnet_50 网络模型
    caffe_root = "/home/nd/caffe_master/"
    net_file = caffe_root + "examples/resnet/resnet_deploy.prototxt"
    model = caffe_root + "examples/resnet/model/resnet_v4_iter_64200.caffemodel"


    caffe.set_mode_cpu()
    net = caffe.Net(net_file, model, caffe.TEST)
    image_mean = np.load(caffe_root + "examples/resnet/furniture_mean.npy").mean(1).mean(1)

    # print 'mean-subtracted values:', zip('RGB', image_mean)
    # 输出结果：mean-subtracted values: [('R', 104.0069879317889), ('G', 116.66876761696767), ('B', 122.6789143406786)]

    data_shape = net.blobs['data'].data.shape
    # print data_shape
    # 输出结果：(10, 3, 224, 224) batch_size:10  channels:3  height:224  weight:224

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', image_mean)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))

    i = 0
    result_f = open("result_test4.txt", 'w')

    for file in os.listdir(image_dir):
        # file_name = file.split('.')[0]
        file_name = file[:-4]
        file_path = os.path.join(image_dir, file)

        image = caffe.io.load_image(file_path)
        transformed_image = transformer.preprocess('data', image)

        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        output_prob = output['prob'][0]
        print 'The predicted class is : ', output_prob.argmax()

        top_1 = output_prob.argsort()[::-1][0]
        top_5 = output_prob.argsort()[::-1][0:10]
        print(type(top_5))
        prob_1 = output_prob[top_1]
        print 'The probabilities is : ', prob_1
        result_str = file_name + " " + str(top_5) + " " + str(prob_1)
        result_f.write(result_str + '\n')

    result_f.close()

    print(i)


if __name__ == '__main__':
    
    main_test()
    
    # main()