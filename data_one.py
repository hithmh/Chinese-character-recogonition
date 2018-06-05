import os
import numpy as np
from PIL import Image as image
import csv
import struct
import cv2
import tensorflow as tf
import matplotlib as plt

train_data_dir = 'F:\\pycharm\\汉字\\train'
test_data_dir = 'F:\\pycharm\\汉字\\test1\\'
words = os.listdir(train_data_dir)
test_filename_list= os.listdir(test_data_dir)
category_num=len(words)
img_size = (256, 256)
datasize= img_size[0] * img_size[1]
def loadOneWord(order):
    path = train_data_dir + '\\'+ words[order] + '\\'
    files = os.listdir(path)
    datas = []
    for file in files:
        file = path + file
        img = np.asarray(image.open(file))
        img = cv2.resize(img, img_size)
        datas.append(img)
    datas = np.array(datas)
    labels = np.zeros([len(datas), len(words)], dtype=np.uint8)
    labels[:, order] = 1
    return datas, labels
def transData():    #将所有数据转存，以后就不用每次都从原始数据读取了
    num = len(words)
    datas = np.array([], dtype=np.uint8)
    datas.shape = -1, 256, 256
    labels = np.array([], dtype=np.uint8)
    labels.shape = -1, 100
    for k in range(num):
        data, label = loadOneWord(k)
        datas = np.append(datas, data, axis=0)
        labels = np.append(labels, label, axis=0)
        print('loading', k)
    np.save('data.npy', datas) #将数据和标签分别存为data和label
    np.save('label.npy', labels)

class DataSet():
    def __init__(self):
        datas = np.load('data.npy')
        labels = np.load('label.npy')
        index = np.arange(0, 40000, 1, dtype=np.int)
        np.random.shuffle(index)
        self.data = datas[index]
        self.label = labels[index]

    def __getitem__(self, index):

        return np.reshape(self.data[index],[-1,256,256,1]).astype('float32'), np.reshape(self.label[index],[-1,100]).astype('float32')

    def __len__(self):
        return len(self.data)

class TestSet():
    def __init__(self,transfer=False):
        self.files = os.listdir(test_data_dir)
        if transfer:
            datas = []
            for file in self.files:
                file = test_data_dir + file
                img = np.asarray(image.open(file))
                img = cv2.resize(img, img_size)
                datas.append(img)

            self.data = np.array(datas)
            np.save('testdata.npy', self.data)
        else:
            self.data = np.load('testdata.npy')
        self.words = os.listdir(train_data_dir)
    def __getitem__(self, index):
        return np.reshape(self.data[index], [-1, img_size[0], img_size[0], 1]).astype('float32')

    def __len__(self):
        return len(self.data)
    def label_data(self,predicted_label):
        i=0

        with open('result.csv','w',encoding='utf-8',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename','label'])
            for file in self.files:
                topfive_posible = np.argsort(-predicted_label[i, :])[0:5]
                writer.writerow([file,self.words[topfive_posible[0]]+self.words[topfive_posible[1]]+self.words[topfive_posible[2]]+self.words[topfive_posible[3]]+self.words[topfive_posible[4]]])
                i=i+1

# def save_csv(test_image_path, predict_label):
#
#     save_arr = np.empty((10000, 2), dtype=np.str)
#     save_arr = pd.DataFrame(save_arr, columns=['filename', 'lable'])
#     predict_label = tran_list2str(predict_label)
#     for i in range(len(test_filename_list)):
#         filename = test_filename_list[i].split('/')[-1]
#         save_arr.values[i, 0] = filename
#         save_arr.values[i, 1] = predict_label[i]
#     save_arr.to_csv('submit_test.csv', decimal=',', encoding='utf-8', index=False, index_label=False)
#     print('submit_test.csv have been write, locate is :', os.getcwd())

def tran_list2str(predict_label):
    string_labels=[]
    for row in range(len(predict_label)):
        index=predict_label[row]
        string_labels.append(words[index])
    return string_labels



    # def transData():    #将所有数据转存，以后就不用每次都从原始数据读取了
    #     num = len(words)
    #     datas = np.array([], dtype=np.uint8)
    #     datas.shape = -1, img_size[0], img_size[1], 1
    #     labels = np.array([], dtype=np.uint8)
    #     labels.shape = -1, 100
    #     for k in range(num):
    #         data, label = LoadFilesInOneWord(k)
    #         datas = np.append(datas, data, axis=0)
    #         labels = np.append(labels, label, axis=0)
    #         print('loading', k)
    #     np.save('data.npy', datas) #将数据和标签分别存为data和label
    #     np.save('label.npy', labels)

    # def LoadFilesInOneWord(k):
    #     path = train_data_dir + '\\' + words[k]
    #     files = os.listdir(path)
    #     datas = []
    #     for file in files:
    #         file_path=path + '\\' + file
    #         img = np.asanyarray(image.open(file_path))
    #         img = cv2.resize(img, img_size)
    #         datas.append(img)
    #     datas = np.array(datas)
    #     datas.shape = len(datas), img_size[0], img_size[1], 1
    #     labels = np.zeros([len(datas), len(words)], dtype = np.uint8)
    #     labels[:, k] = 1
    #     return datas, labels
    #
    # class TrainSet():
    #     def __init__(self):
    #         datas = np.load('data.npy')
    #         labels = np.load('label.npy')
    #         index = np.arange(0, len(datas), 1, dtype=np.int)
    #         np.random.shuffle(index)
    #         self.data = datas[index]
    #         self.label = labels[index]
    #
    #     def __getitem__(self, index):
    #         return self.data[index], self.label[index]
    #
    #     def __len__(self):
    #         return len(self.data)