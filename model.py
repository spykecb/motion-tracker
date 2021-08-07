import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, ops
import torch.nn.functional as F
from PIL import Image
import numpy as np
import pandas as pd
import os

class MotionDataset(Dataset):
    """Motion dataset."""

    def __init__(self, csv_file_path, root_dir, transforms, minmax, minmax_z, img_size = 512, bmodel = None):        
        self.csv_file = pd.read_csv(csv_file_path, header=None)
        self.root_dir = root_dir
        self.transforms = transforms
        self.minmax = minmax
        self.minmax_z = minmax_z
        self.bmodel = bmodel
        self.img_size = img_size

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(
            self.root_dir, 
            self.csv_file.iloc[idx, 0])

        
        # print(img_name)
        image = Image.open(img_name)
        
        input_to = 3
        positions_to = input_to + 22 * 2
        confidences_to = positions_to + 22
        boundaries_to = confidences_to + 4
        details = self.csv_file.iloc[idx, 1:input_to].astype('float32').values
        positions = self.csv_file.iloc[idx, input_to:positions_to].astype('float32').values
        confidences = self.csv_file.iloc[idx, positions_to:confidences_to].astype('float32').values
        boundaries = self.csv_file.iloc[idx, confidences_to:boundaries_to].astype('float32').values

        # #normalization
        positions = (positions - self.minmax[0]) / (self.minmax[1] - self.minmax[0])
        boundaries = boundaries / 512.0

        # TODO: input positions have inverted y (lower left = (0,0)), it should be upper left
        positions = np.array(positions)
        positions[1::2] = 1 - positions[1::2]
        confidences = np.array(confidences)
        boundaries = np.array(boundaries)

        #try label confidences instead
        # confidences = confidences * (np.arange(22) + 1)

        #finding bounding box
        if self.bmodel is not None:
            # orig_img = transforms.ToTensor()(image).view(-1, 3, self.img_size,self.img_size)
            # bbox_output = self.bmodel.forward(orig_img.to('cuda'))[0]

            # CHEATING
            bbox_output = boundaries
            rect = tuple(int(b * 512) for b in bbox_output)
            image = image.crop(rect)
            pass
        
        if self.transforms:
            image = self.transforms(image)
        image = np.array(image)

        inp = {}
        inp["images"] = image
        inp["details"] = details
        inp["bboxes"] = np.array([])
        if self.bmodel is not None:
            inp["bboxes"] = bbox_output
            # positions = align_targets_in_bounding_boxes(positions, bbox_output, 256)
            pass
        output = {}
        output["positions"] =  positions.reshape(-1,2)
        output["confidences"] = confidences
        output["boundaries"] = boundaries

        assert positions.min() >= 0
        assert positions.max() <= 1
        
        return inp, output

class PositionFinder(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        #defining the layers here

        #Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, 1, padding=0)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 1, padding=0)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        # self.roi_size = 16
        # self.roi = ops.PSRoIAlign((self.roi_size,self.roi_size), 1/16, 2)

        size = self.img_size//16 * self.img_size//16
        #Linear layers
        self.hidden1 = nn.Linear(512*size, 1000)
        # self.hidden1 = nn.Linear(512//(self.roi_size*self.roi_size)*size, 500)
        # self.hidden2 = nn.Linear(1024, 512)
        self.output = nn.Linear(1000, 22*3)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.4)

    def initialize(self):
        self.hidden1.weight.data.zero_()
        self.hidden1.bias.data.zero_()
        self.hidden2.weight.data.zero_()
        self.hidden2.bias.data.zero_()
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
        pass
        
    def forward(self, x, details, bboxes):
        #Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        # rois = []
        # for i in range(len(bboxes)):
        #     rois.append([i, bboxes[i][0]-0.5 * 16, bboxes[i][1]-0.5 * 16, bboxes[i][2]+0.5 * 16, bboxes[i][3]+0.5 * 16])
        # rois = torch.Tensor(rois).to('cuda')
        # print("before", x[0])
        # x = F.relu(self.roi(x, rois))
        # print("after", x[0])

        # make sure input tensor is flattened
        # torch.Size([16, 512, 16, 16])
        size = self.img_size//16 * self.img_size//16
        x = x.view(-1, 512*size)
        # x = x.view(-1, 512//(self.roi_size*self.roi_size)*size)
        # x = torch.cat((x,details, bboxes), dim = 1)
        

        #Forward pass through the network, returns the output
        x = self.dropout(F.relu(self.hidden1(x)))
        # x = self.dropout(F.relu(self.hidden2(x)))
        x = self.output(x)

        #Bounding boxes
        for i in range(x.shape[0]):
            w = (bboxes[i][2] - bboxes[i][0]) * self.img_size
            h = (bboxes[i][3] - bboxes[i][1]) * self.img_size
            x[i,0::2] = bboxes[i][0] + x[i,0::2]*(w/self.img_size)
            x[i,1::2] = bboxes[i][1] + x[i,1::2]*(h/self.img_size)

        #confidences
        y = x[:,22*2:22*3]
        x = x[:,:22*2]
        x = x.view(x.shape[0], -1, 2)
        y = torch.sigmoid(y)
        # y = F.log_softmax(y, dim=1)
        return x, y

class BoundingBoxFinder(nn.Module):
    def __init__(self, img_size):
        super().__init__()
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

def align_targets_in_bounding_boxes(output, bboxes, img_size = 256):
    x = bboxes[0] 
    y = bboxes[1]
    w = (bboxes[2] - x) * img_size # denormalized width of bounding box
    h = (bboxes[3] - y) * img_size # denormalized height of bounding box
    # print(x * img_size,y * img_size,w,h)
    # print(w,h, "before", output)
    # print(output.shape)
    output[0::2] = (output[0::2] - x.item()) * (img_size / w.item())
    output[1::2] = (output[1::2] - y.item()) * (img_size / h.item())
    # print("after", output)
    output = np.clip(output, 0, 1)

    return output

def crop_image(image, x1,y1,x2,y2):
    return image[:,x1:x2,y1:y2]

