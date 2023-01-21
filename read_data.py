import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
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
    label_list = [int(i) for i in label_list] 
    
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
    label_list = [int(i) for i in label_list] 
    
    return  image_list, label_list
    #返回两个list 分别为图片文件名及其标签  顺序已被打乱

class liquidDataset(Data.Dataset):
    def __init__(self, image_paths, labels = None, train = True, test = False):
        self.paths = image_paths
        self.test = test
        #if self.test == False:
        self.labels = labels
        self.train = train
        self.default_rgb_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.48877397, 0.47886574, 0.45613098), (0.04925239, 0.06920359, 0.10496992))]) #normalized for pretrained network
        self.default_depth_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((414.76987), (171.60141))]) #normalized for pretrained network

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        #print(self.paths[i])
        RGB_path=self.paths[i].replace('depth','RGB')
        #print(RGB_path)
        RGB_image = io.imread(RGB_path) #load to rgb
        depth_image = io.imread(self.paths[i]) #load to depth
        depth_image = depth_image.astype(np.float32)
        #image=transform.resize(image, (128, 128))*255
        with open(RGB_path.replace('png','txt'),'r') as f:    
            string = f.read()
            bbox=[float(i) for i in string.split(' ')[1:5]]
        x1=(bbox[0]-bbox[2]/2.)*RGB_image.shape[1]
        y1=(bbox[1]-bbox[3]/2.)*RGB_image.shape[0]
        x2=(bbox[0]+bbox[2]/2.)*RGB_image.shape[1]
        y2=(bbox[1]+bbox[3]/2.)*RGB_image.shape[0]
        box=torch.tensor([[x1,y1,x2,y2]])

        label = torch.tensor([self.labels[i]])
        RGB_image = self.default_rgb_transform(RGB_image)
        depth_image = self.default_depth_transform(depth_image)
        depth_image = depth_image.squeeze(2)
        
        return RGB_image, depth_image, box, label

if __name__ == '__main__':
    image_list, label_list=get_data()
    print(image_list[:5])
    print(label_list[:5])
    image_list, label_list=get_files(image_list, label_list)
    print(image_list[:5])
    print(label_list[:5])
