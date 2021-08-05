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

    def __init__(self, csv_file_path, root_dir, transforms, minmax, minmax_z, bboxes = None):        
        self.csv_file = pd.read_csv(csv_file_path, header=None)
        self.root_dir = root_dir
        self.transforms = transforms
        self.minmax = minmax
        self.minmax_z = minmax_z
        self.bboxes = bboxes

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(
            self.root_dir, 
            self.csv_file.iloc[idx, 0])
        image = Image.open(img_name)
        # if self.bboxes:
        #     image = image.crop(tuple(int(b * 512) for b in self.bboxes[idx]))
        
        input_to = 3
        positions_to = input_to + 22 * 2
        confidences_to = positions_to + 22
        boundaries_to = confidences_to + 4
        details = self.csv_file.iloc[idx, 1:input_to].astype('float32').values
        positions = self.csv_file.iloc[idx, input_to:positions_to].astype('float64').values
        confidences = self.csv_file.iloc[idx, positions_to:confidences_to].astype('float32').values
        boundaries = self.csv_file.iloc[idx, confidences_to:boundaries_to].astype('float32').values

        # #normalization
        positions = (positions - self.minmax[0]) / (self.minmax[1] - self.minmax[0])
        boundaries = boundaries / 512.0

        positions = np.array(positions)
        confidences = np.array(confidences)
        boundaries = np.array(boundaries)
        
        if self.transforms:
            image = self.transforms(image)
        image = np.array(image)

        inp = {}
        inp["images"] = image
        inp["details"] = details
        if self.bboxes:
            inp["bboxes"] = self.bboxes[idx]
        output = {}
        output["positions"] = positions
        output["confidences"] = confidences
        output["boundaries"] = boundaries
        
        
        return inp, output

class PositionFinder(nn.Module):
    def __init__(self, img_size, device):
        super().__init__()
        self.device = device
        self.img_size = img_size
        #defining the layers here

        #Convolutional layers
        #TODO:maybe it would be wiser to change the kernel size from 3 since we're using 512x512 images. Should be slower though.
        self.conv1 = nn.Conv2d(3, 64, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)

        size = self.img_size//16 * self.img_size//16
        #Linear layers
        self.hidden1 = nn.Linear(512*size + 6, 1024)
        # self.hidden2 = nn.Linear(1000, 500)
        self.output = nn.Linear(1024, 22*2)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.4)

    def initialize(self):
        self.hidden1.weight.data.zero_()
        self.hidden1.bias.data.zero_()
        # self.hidden2.weight.data.zero_()
        # self.hidden2.bias.data.zero_()
        self.conv1.weight.data.zero_()
        self.conv1.bias.data.zero_()
        self.conv2.weight.data.zero_()
        self.conv2.bias.data.zero_()
        self.conv3.weight.data.zero_()
        self.conv3.bias.data.zero_()
        self.conv4.weight.data.zero_()
        self.conv4.bias.data.zero_()
        self.output.weight.data.zero_()
        self.output.bias.data.zero_()
        
    def forward(self, x, details, bboxes):
        x, details, bboxes = x.to(self.device), details.to(self.device), bboxes.to(self.device)
        #Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # make sure input tensor is flattened
        size = self.img_size//16 * self.img_size//16
        x = x.view(-1, 512*size)
        x = torch.cat((x,details, bboxes), dim = 1)

        #Forward pass through the network, returns the output
        x = self.dropout(F.relu(self.hidden1(x)))
        # x = self.dropout(F.relu(self.hidden2(x)))
        x = self.output(x)

        #Add bbox data
        #TODO: since the images are resized, maybe adding to the linear nn layer is better
        # for i in range(len(x)):
        #     x[i][0::2] = x[i][0::2] + bboxes[i][0]
        #     x[i][1::2] = x[i][1::2] + bboxes[i][1]
        # y = x[:,22*2:]
        # x = x[:,:22*2]
        # y = torch.sigmoid(y)
        return x

class BoundingBoxFinder(nn.Module):
    def __init__(self, img_size, device):
        super().__init__()
        self.device = device
        self.img_size = img_size
        #defining the layers here

        #Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)

        size = self.img_size//8 * self.img_size//8
        #Linear layers
        self.hidden1 = nn.Linear(256*size, 512)
        # self.hidden2 = nn.Linear(1000, 500)
        self.output = nn.Linear(512, 4)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.4)

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
        
    def forward(self, x):
        x = x.to(device)
        #Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # make sure input tensor is flattened
        size = self.img_size//8 * self.img_size//8
        x = x.view(-1, 256*size)

        #Forward pass through the network, returns the output
        x = self.dropout(F.relu(self.hidden1(x)))
        x = self.output(x)
        return x


