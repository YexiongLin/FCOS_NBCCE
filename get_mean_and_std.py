import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import *
from albumentations import *
from albumentations.pytorch import ToTensor
from skimage import transform 
import json
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def tobin(x,lenth=10):
    x=int(x)
    y=bin(x)
    y=y.split('b')[1]
    result=[]
    for i in range(lenth-len(y)):
        result.append(0.0)
    for i in y:
        result.append(float(i))
    return result

def get_data(file_dir='./data'):
    data = []
    label = []

    for i in os.listdir(file_dir):
        for j in os.listdir(file_dir+'/'+i):
            data.append(file_dir+'/'+i+'/'+j+'/depth')
            label.append(j)
    
    #利用shuffle打乱顺序
    temp = np.array([data, label])
    temp = temp.transpose()
    #print(temp.shape)
    np.random.shuffle(temp)
 
    #从打乱的temp中再取出list（img和lab）
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [tobin(i) for i in label_list] 
    
    return  image_list, label_list
    #返回两个list 分别为图片文件名及其标签  顺序已被打乱

def get_files(all_data,all_label):
    data = []
    label = []

    for i in range(len(all_data)):
        for j in os.listdir(all_data[i]):
            data.append(all_data[i]+'/'+j)
            label.append(all_label[i])
    
    #利用shuffle打乱顺序
    temp = np.array([data, label])
    temp = temp.transpose()
    #print(temp.shape)
    np.random.shuffle(temp)
 
    #从打乱的temp中再取出list（img和lab）
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    
    return  image_list, label_list
    #返回两个list 分别为图片文件名及其标签  顺序已被打乱

class liquidDataset(Data.Dataset):
    def __init__(self, image_paths, labels = None, train = True, test = False):
        self.paths = image_paths
        self.test = test
        #if self.test == False:
        self.labels = labels
        self.train = train
        self.train_transform = Compose([HorizontalFlip(p=0.5),
                                  VerticalFlip(p=0.5),
                                  ShiftScaleRotate(rotate_limit=25.0, p=0.7),
                                  OneOf([IAAEmboss(p=1),
                                         IAASharpen(p=1),
                                         Blur(p=1)], p=0.5),
                                  IAAPiecewiseAffine(p=0.5)])
        self.test_transform = Compose([HorizontalFlip(p=0.5),
                                       VerticalFlip(p=0.5),
                                       ShiftScaleRotate(rotate_limit=25.0, p=0.7)])
        self.default_rgb_transform = Compose([ToTensor()]) #normalized for pretrained network
        self.default_depth_transform = Compose([ToTensor()]) #normalized for pretrained network

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        #print(self.paths[i])
        RGB_path=self.paths[i].replace('depth','RGB')
        RGB_image = io.imread(RGB_path) #load to rgb
        depth_image = io.imread(self.paths[i]) #load to depth
        #image=transform.resize(image, (128, 128))*255

        label = torch.tensor(self.labels[i])
        RGB_image = self.default_rgb_transform(image=RGB_image)['image']
        depth_image = self.default_depth_transform(image=depth_image)['image']
        depth_image=depth_image.unsqueeze(0)

        return RGB_image, depth_image, label

if __name__ == '__main__':
    image_list, label_list=get_data()
    print(image_list[:5])
    print(label_list[:5])
    image_list, label_list=get_files(image_list, label_list)
    print(image_list[:5])
    print(label_list[:5])
    train_data = liquidDataset(image_list, label_list)
    train_loader = DataLoader(dataset=train_data, batch_size=500, shuffle=True) # # 500张图片的mean std
    print(len(iter(train_loader).next()))
    train = iter(train_loader).next()[0]  # 500张图片的mean std
    print(train.shape)
    print(train[0])
    train_mean = np.mean(train.numpy(), axis=(0, 2, 3))
    train_std = np.std(train.numpy(), axis=(0, 2, 3))
    print(train_mean)
    print(train_std)
