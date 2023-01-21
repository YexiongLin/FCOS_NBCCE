import numpy as np
import pandas as pd
from read_data import *
from sklearn.model_selection import train_test_split
import random

np.random.seed(0)
random.seed(0)

datas, labels=get_data()
train_datas, test_datas, train_labels, test_labels=train_test_split(datas, labels,test_size=0.2)
trainsets=np.array([train_datas,train_labels]).T
testsets=np.array([test_datas,test_labels]).T
df0=pd.DataFrame(data=trainsets)
df0.to_csv('trainsets.csv',index=False)
df1=pd.DataFrame(data=testsets)
df1.to_csv('testsets.csv',index=False)