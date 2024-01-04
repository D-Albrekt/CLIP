import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class visa_dataset(Dataset):
    def __init__(self, csv_file_path, data_path, mode = 'test', transform = None):
        self.label_frame = pd.read_csv(csv_file_path)
        self.data_dir = data_path
        self.transform = transform

        self.types = self.label_frame.object[:].unique()

        if mode == 'test':
            self.label_frame = self.label_frame[self.label_frame['split'] =='test']
        elif mode == 'train':
            self.label_frame = self.label_frame[self.label_frame['split'] =='train']
        self.size = len(self.label_frame)
        self.img_paths = self.label_frame['image']
        self.labels = self.label_frame['label']
        self.types = self.label_frame['object']
        self.masks = self.label_frame['mask']
        print(self.masks)
    def __length__(self):
        return self.size
    def __getitem__(self, idx):
        return "hej"

        #self.img_paths, self.gt_paths, self.labels, self.types
    
    