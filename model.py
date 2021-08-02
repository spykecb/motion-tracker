import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import numpy as np
import pandas as pd
import os

class MotionDataset(Dataset):
    """Motion dataset."""

    def __init__(self, csv_file_path, root_dir, transforms, minmax, minmax_z):        
        self.csv_file = pd.read_csv(csv_file_path)
        self.root_dir = root_dir
        self.transforms = transforms
        self.minmax = minmax
        self.minmax_z = minmax_z

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(
            self.root_dir, 
            self.csv_file.iloc[idx, 0])
        image = Image.open(img_name)
        
        height = self.csv_file.iloc[idx, 1].astype('float32')
        transform = self.csv_file.iloc[idx, 2:].astype('float32').values
        
        #normalization
        helper_arr = (np.arange(transform.shape[0])) % 3 == 2
        helper_arr_neg = ~helper_arr 
        # print(transform[helper_arr_neg] )
        transform[helper_arr_neg] = (transform[helper_arr_neg] - self.minmax[0]) / (self.minmax[1] - self.minmax[0])


        #depth normalized to similar range as screen position
        transform[helper_arr] = (transform[helper_arr] - self.minmax_z[0])*(self.minmax_z[1] - self.minmax_z[0])
        print(transform[helper_arr_neg] )
        print(transform[helper_arr] )
        # transform = sigmoid(transform)
        
        transform = np.array(transform)
        
        if self.transforms:
            image = self.transforms(image)
        image = np.array(image)
        
        return image, height, transform

class Network(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        #defining the layers here

        #Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)

        size = self.img_size//8 * self.img_size//8
        #Linear layers
        self.hidden1 = nn.Linear(256*size + 1, 1000)
        # self.hidden2 = nn.Linear(n_hidden_units, n_hidden_units//2)
        self.output = nn.Linear(1000, 22*3)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def initialize(self):
        self.hidden1.weight.data.zero_()
        self.hidden1.bias.data.zero_()
        self.conv1.weight.data.zero_()
        self.conv1.bias.data.zero_()
        self.conv2.weight.data.zero_()
        self.conv2.bias.data.zero_()
        self.conv3.weight.data.zero_()
        self.conv3.bias.data.zero_()
        self.output.weight.data.zero_()
        self.output.bias.data.zero_()
        
    def forward(self, x, height):
        #Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # make sure input tensor is flattened
        size = self.img_size//8 * self.img_size//8
        x = x.view(-1, 256*size)
        x = torch.cat((x,height.view(-1, 1)), dim = 1)

        #Forward pass through the network, returns the output
        x = self.dropout(F.relu(self.hidden1(x)))
        # x = self.dropout(F.relu(self.hidden2(x)))
        x = torch.sigmoid(self.output(x)).squeeze(1)
        # x = F.log_softmax(x, dim=1)
        return x



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

