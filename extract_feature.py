# coding=utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import csv
import glob
import sys
import caffe
from scipy.spatial.distance import pdist
import time
import argparse

# dist = pdist(np.vstack([x, y, z]), 'cosine')  # pdist计算向量之间的余弦距离

# caffe_root = "/home/nd/caffe_master/"
# net_file = caffe_root + "examples/resnet/resnet_deploy.prototxt"
# weight_file = caffe_root + "examples/resnet/model/resnet_v4_iter_64200.caffemodel"
# caffe.set_mode_cpu()
# net = caffe.Net(net_file, weight_file, caffe.TEST)

# image_mean = np.load(caffe_root + "examples/resnet/furniture_mean.npy").mean(1).mean(1)
# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_transpose('data', (2, 0, 1))
# transformer.set_mean('data', image_mean)
# transformer.set_raw_scale('data', 255)
# transformer.set_channel_swap('data', (2, 1, 0))

def parse_args():
    parser = argparse.ArgumentParser(description='calssify model test')
    # general
    parser.add_argument('--caffe_root', default="/home/nd/caffe_master/", help='')
    parser.add_argument('--net_file', default='examples/resnet/resnet_deploy.prototxt', help='')
    parser.add_argument('--model_file', default='examples/resnet/model/resnet_v4_iter_64200.caffemodel', help='')
    parser.add_argument('--mean_file', default='examples/resnet/furniture_mean.npy', help='')
    parser.add_argument('--feature_dir', default="./feature/", help='')
    args = parser.parse_args()
    return args


def load_model():
    args = parse_args()
    global net, image_mean
    net_file = args.caffe_root + args.net_file
    model_file = args.caffe_root + args.model_file
    mean_file = args.caffe_root + args.mean_file
    caffe.set_mode_gpu()
    net = caffe.Net(net_file, model_file, caffe.TEST)
    image_mean = np.load(mean_file).mean(1).mean(1)
    return args, net, image_mean


def cos_sim_mat(Matrix, B):
    """计算矩阵中每个行向量与另一个行向量的余弦相似度"""
    num = np.dot(Matrix, B.T)
    denom = np.linalg.norm(Matrix, axis=1, keepdims=True) * np.linalg.norm(B)
    # denom = denom.reshape(-1)
    cos_val = num / denom
    sim = 0.5 + 0.5 * cos_val
    return sim


def extract_feature(subimg):
    """extract feature of image"""
    image = caffe.io.load_image(subimg)
    global net, image_mean
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', image_mean)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    # print(output['prob'].shape)
    feature = net.blobs['pool5'].data[0]
    feature_reshape = feature.reshape((1, 2048))

    return feature_reshape


def extract_feature_dataset(image_path, feature_path):
    feature_file = os.path.join(feature_path, "furniture_feature.txt")
    furniture_calss_file = os.path.join(feature_path, "furniture_class.txt")

    if os.path.exists(feature_file) and os.path.exists(furniture_calss_file):
        print("file exist")
        furniture_feature = np.loadtxt(feature_file)
        with open(furniture_calss_file, 'r') as rf:
            data = rf.readlines()
        furniture_id_list = eval(data[0].strip())

        if furniture_feature.shape[0] == len(furniture_id_list) != 0:
            i = 1
            print(1)
        else:
            i = 0
            print(0)
    else:
        furniture_id_list = []
        i = 0
        print("file not exist")

    for sub_dir in os.listdir(image_path):
        print(sub_dir)
        furniture_img_dir = os.path.join(image_path, sub_dir)
        for file in os.listdir(furniture_img_dir):
            file_path = os.path.join(furniture_img_dir, file)

            i += 1
            feature_reshape = extract_feature(file_path)
            f = feature_reshape

            furniture_id_list.append(sub_dir)

            if i == 1:
                furniture_feature = f
            else:
                furniture_feature = np.vstack((furniture_feature, feature_reshape))

    np.savetxt(feature_file, furniture_feature)
    with open(furniture_calss_file, 'w') as f:
        f.write(str(furniture_id_list))

    print("Done")
    return furniture_feature, furniture_id_list


def feature_KMeans(feature, n_clusters=10):
    """对数据向量进行聚类"""
    from sklearn.cluster import KMeans

    estimator = KMeans(n_clusters)
    estimator.fit(feature)
    label_pred = estimator.labels_
    centroids = estimator.cluster_centers_
    inertia = estimator.inertia_

    return label_pred, centroids


def extract_feature_KMeans(image_path, feature_path):
    feature_kmeans_file = os.path.join(feature_path, "furniture_feature_kmeans.txt")
    furniture_calss_file = os.path.join(feature_path, "furniture_class_kmeans.txt")

    if os.path.exists(feature_kmeans_file) and os.path.exists(furniture_calss_file):
        print("file exist")
        feature_kmeans = np.loadtxt(feature_kmeans_file)
        with open(furniture_calss_file, 'r') as rf:
            data = rf.readlines()
        furniture_id_list = eval(data[0].strip())

        if feature_kmeans.shape[0] == len(furniture_id_list) != 0:
            print(1)
        else:
            print(0)
    else:
        furniture_id_list = []
        feature_kmeans = None
        print("file not exist")

    sub_dir_list = sorted(os.listdir(image_path))
    for sub_dir in sub_dir_list:
        if not os.path.isdir(os.path.join(image_path, sub_dir)):
            print(sub_dir +"is not dir")
            continue
        if sub_dir in furniture_id_list:
            continue
        print(sub_dir)
        furniture_img_dir = os.path.join(image_path, sub_dir)

        if len(os.listdir(furniture_img_dir)) < 10:
            n_clusters = len(os.listdir(furniture_img_dir))
        else:
            n_clusters = 10

        i = 0
        for file in os.listdir(furniture_img_dir):
            file_path = os.path.join(furniture_img_dir, file)

            i += 1
            feature_reshape = extract_feature(file_path)
            f = feature_reshape

            if i == 1:
                furniture_feature = f
            else:
                furniture_feature = np.vstack((furniture_feature, feature_reshape))

        label_pred, sub_feature_kmeans = feature_KMeans(furniture_feature, n_clusters)

        if feature_kmeans is None:
            feature_kmeans = sub_feature_kmeans
        else:
            feature_kmeans = np.vstack((feature_kmeans, sub_feature_kmeans))

        furniture_id_list.extend([sub_dir for i in range(n_clusters)])

    np.savetxt(feature_kmeans_file, feature_kmeans)

    with open(furniture_calss_file, 'w') as f:
        f.write(str(furniture_id_list))

    print("Done")
    print(len(furniture_id_list))
    print(feature_kmeans.shape)
    return feature_kmeans, furniture_id_list


def load_feature(feature_path):
    """load features and furniture id"""
    feature_file = os.path.join(feature_path, "furniture_feature_kmeans.txt")
    furniture_calss_file = os.path.join(feature_path, "furniture_class_kmeans.txt")
    furniture_feature = np.loadtxt(feature_file)
    with open(furniture_calss_file, 'r') as rf:
            data = rf.readlines()
    furniture_id_list = eval(data[0].strip())

    return furniture_feature, furniture_id_list


def furniture_calssify(img, furniture_id_list, furniture_feature, top_n=10):
    """"""
    if isinstance(top_n, int):
        pass
    else:
        top_n = int(top_n)

    key = "top_{}".format(str(top_n))
    result = {}
    start1 = time.time()
    feature_reshape = extract_feature(img)
    print("extract_feature time", time.time()-start1)
    sim = cos_sim_mat(furniture_feature, feature_reshape)
    sim = sim.reshape((1, -1))
    idx = np.argsort(-sim)
    # print(idx)
    top_n_sim = []
    top_n_result = []
    for i in range(idx[0].size):
        # print(idx[0][i])
        if i == 0:
            result["maxSimilarity"] = str(sim[0][idx[0][i]])

        furniture_id = furniture_id_list[idx[0][i]]
        if furniture_id not in top_n_sim and len(top_n_sim) < top_n:
            top_n_sim.append(furniture_id)
            top_n_result.append((furniture_id, "%f" %(sim[0][idx[0][i]])))

    result[key] = top_n_result
    return result, top_n_sim


def test_accuracy(furniture_feature, furniture_id_list):
    '''计算测试集中top1, top2 ... top10的准确度'''
    root_path = "./test_data_231_0.6/"

    save_result_file = "save_test_result_10_977_bg_0.6.txt"
    res_f = open(save_result_file, 'w')

    top_10 = [0] * 10
    file_num = 0
    for sub_dir in sorted(os.listdir(root_path)):
        print(sub_dir)
        
        # if sub_dir.startswith("00"):
        #     continue

        dir_path = os.path.join(root_path, sub_dir)
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            result, top_n_sim = furniture_calssify(file_path, furniture_id_list, furniture_feature, top_n=10)
            res_f.write(sub_dir+'/'+file+' ' + str(top_n_sim) + '\n')
            file_num += 1
            for i in range(1, 11):
                if sub_dir in top_n_sim[0:i]:
                    top_10[i-1] += 1
                else:
                    continue
        print(top_10)

    print(top_10)              
    # arr_top_10 = np.array(top_10)
    top_10_acc = map(lambda x:x/float(file_num), top_10)
    res_f.write('\n' + str(top_10) + '\n' + str(file_num) + '\n' + str(top_10_acc) + '\n')

    res_f.close()


if __name__ == '__main__':
    """"""
    image_path = "/home/nd/datasets/furniture_bg/"
    feature_path = "./feature/"

    load_model()
    print("loadind model...")

    # furniture_feature, furniture_id_list = extract_feature_dataset(image_path, feature_path)
    # feature_kmeans, furniture_id_list = extract_feature_KMeans(image_path, feature_path)

    start = time.time()
    furniture_feature, furniture_id_list = load_feature(feature_path)
    print(time.time() - start)
    print("Feature load Done.")


    test_accuracy(furniture_feature, furniture_id_list)


    # image_dir = "./57764_error/"
    # result_f = open("57764_result_sub.txt", 'w')
    # err_result_f = open("57764_error_sub.txt", 'w')
    # import shutil

    # for file in os.listdir(image_dir):
    #     print(file)
    #     file_path = os.path.join(image_dir, file)
    #     result, top_n_sim = furniture_calssify(file_path, furniture_id_list, furniture_feature, top_n=10)
    #     if "00126" not in top_n_sim:           
    #         err_result_f.write(file + "  " + str(top_n_sim) + "\n")
    #         error_path = os.path.join("./error_file_sub/", file)
    #         shutil.copyfile(file_path, error_path)

    #     result_f.write(file + "  " + str(top_n_sim) + "\n")

    # result_f.close()

'''
    img_file = "./test_image/*.png"

    wf = open("classify_result_2.txt", 'w')
    for file in glob.glob(img_file):
        t = time.time()
        top_10_id, top_10_prob = furniture_calssify(file, furniture_id_list, furniture_feature)
        print(time.time() - t)
        print(top_10_id)
        # print(top_10_prob)

        wf.write(file + '\n')
        wf.write(str(top_10_id) + '\n')

    wf.close()
'''





# imgdir = "./jiaju_test/*.png"
# feature_ndarray = np.random.randn(1, 2048) # 生成一个随机的feature_ndarray
# imglist = []
# for subimg in glob.glob(imgdir):
#     imglist.append(subimg)
#     image = caffe.io.load_image(subimg)
#     transformed_image = transformer.preprocess('data', image)
#     net.blobs['data'].data[...] = transformed_image
#     output = net.forward()
#     # print(output['prob'].shape)
#     feature = net.blobs['pool5'].data[0]
#     feature_reshape = feature.reshape((1, 2048))

#     feature_ndarray = np.r_['0', feature_ndarray, feature_reshape]
#     print(feature_ndarray.shape)

# final_feature = feature_ndarray[1:]
# print(type(final_feature))
# feature_num = len(final_feature)

# for img1, img2 in itertools.combinations(range(feature_num), 2):

#     v1 = final_feature[img1].reshape(1, 2048)
#     v2 = final_feature[img2].reshape(1, 2048)

#     cos_distance = np.sum((v1*v2), 1) / (np.linalg.norm(v1)*np.linalg.norm(v2))
#     print(cos_distance)

# print(imglist)
