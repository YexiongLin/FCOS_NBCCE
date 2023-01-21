import os
import pandas as pd

def get_files(all_data,all_label):
    data = []
    label = []

    for i in range(len(all_data)):
        for j in os.listdir(all_data[i]):
            data.append(all_data[i]+'/'+j)
            label.append(all_label[i])
    
    return  data, label

test_datas, test_labels=pd.read_csv('trainsets.csv').iloc[:,:].values.T
test_datas, test_labels=get_files(test_datas, test_labels)