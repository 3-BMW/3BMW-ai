# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

import json

def label2num(df):
    with open('./data/label2num.json', 'r') as f:
        label2num= json.load(f)
    df['label']= df['label'].apply(lambda x: int(label2num[x]))

    return df

def num2label(df):
    with open('./data/num2label.json', 'r') as f:
        num2label= json.load(f)
    df['label']= df['label'].apply(lambda x: num2label[x])

    return df

class TrainDataset(Dataset):
    def __init__(self, df, transforms= None):
        self.data= df
        self.transforms= transforms
        # self.get_labels= self.data.loc[:, 'label'].tolist

    def __getitem__(self, idx):
        # print(list(self.data.iloc[idx]['path']))
        image= Image.open(self.data.iloc[idx]['path'])
        label= self.data.iloc[idx]['label']

        if self.transforms:
            image= self.transforms(image)

        return image, label
    
    def __len__(self):
        return len(self.data['label'])

if __name__ == '__main__':

    PATH= './data/train.csv'

    test_dataset= TrainDataset(PATH)
    print(test_dataset[3])