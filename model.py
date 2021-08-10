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

        #only caring about head,hand,and legs
        positions = np.concatenate((positions[10:12], positions[18:20], positions[34:36], positions[24:26], positions[40:42]))
        confidences = np.array([confidences[5], confidences[9], confidences[17], confidences[12], confidences[20]])

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
            orig_img = transforms.ToTensor()(image).view(-1, 3, self.img_size,self.img_size)
            bbox_output = self.bmodel.forward(orig_img.to('cuda'))[0]

            # CHEATING
            # bbox_output = boundaries
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
            positions = align_targets_in_bounding_boxes(positions, bbox_output, 256)
            pass
        output = {}
        # output["positions"] =  positions
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
        self.conv1 = nn.Conv2d(3, 16, 1, padding=0)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 1, padding=0)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2,2)
        # self.roi_size = 16
        # self.roi = ops.PSRoIAlign((self.roi_size,self.roi_size), 1/16, 2)

        size = self.img_size//16 * self.img_size//16
        #Linear layers
        self.hidden1 = nn.Linear(128*size + 2, 500)
        self.dense1_bn = nn.BatchNorm1d(1000)
        # self.hidden1 = nn.Linear(512//(self.roi_size*self.roi_size)*size, 500)
        # self.hidden2 = nn.Linear(1024, 512)
        self.output = nn.Linear(500, 5*2)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

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
        
    def forward(self, x, details, bboxes):
        # Convolutional layers
        x = self.conv1_bn(F.relu(self.pool(self.conv1(x))))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = F.relu(self.pool(self.conv4(x)))

        # make sure input tensor is flattened
        # torch.Size([16, 512, 16, 16])
        size = self.img_size//16 * self.img_size//16
        x = x.view(-1, 128*size)
        # x = x.view(-1, 512//(self.roi_size*self.roi_size)*size)
        x = torch.cat((x,details), dim = 1)
        # x = torch.cat((x,details, bboxes), dim = 1)
        

        #Forward pass through the network, returns the output
        x = self.dropout(F.relu(self.hidden1(x)))
        # x = self.dropout(F.relu(self.hidden2(x)))
        x = self.output(x)

        #confidences
        # y = x[:,5*2:5*3]
        # x = x[:,:5*2]
        x = x.view(x.shape[0], -1, 2)
        # y = torch.sigmoid(y)
        # y = F.log_softmax(y, dim=1)
        return x

class BoundingBoxFinder(nn.Module):
    def __init__(self, img_size, param):
        super().__init__()
        self.img_size = img_size
        self.param = param
        #defining the layers here

        #Convolutional layers
        self.conv1 = nn.Conv2d(3, param["conv1_c"], 1, padding=0)
        self.conv1_bn = nn.BatchNorm2d(param["conv1_c"])
        self.conv2 = nn.Conv2d(param["conv1_c"], param["conv2_c"], 3, padding=1)
        self.conv3 = nn.Conv2d(param["conv2_c"], param["conv3_c"], 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)

        size = self.img_size//8 * self.img_size//8
        #Linear layers
        self.hidden1 = nn.Linear(param["conv3_c"]*size, param["hidden"])
        # self.hidden2 = nn.Linear(1000, 500)
        self.output = nn.Linear(param["hidden"], 4)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=param["drop"])

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
        x = F.relu(self.pool(self.conv1(x)))
        if self.param["conv1_bn"]:
            x = self.conv1_bn(x)
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))

        # make sure input tensor is flattened
        size = self.img_size//8 * self.img_size//8
        x = x.view(-1, self.param["conv3_c"]*size)

        #Forward pass through the network, returns the output
        x = self.dropout(F.relu(self.hidden1(x)))
        x = self.output(x)
        
        return x

def align_targets_in_bounding_boxes(output, bboxes, img_size = 256):
    res = np.copy(output)
    x = bboxes[0] 
    y = bboxes[1]
    w = (bboxes[2] - x) # denormalized width of bounding box
    h = (bboxes[3] - y) # denormalized height of bounding box
    # print(x * img_size,y * img_size,w,h)
    # print(w,h, "before", output)
    # print("({} - {}) * ({}/{})".format(output[0::2],x.item(),img_size,w.item()))
    res[0::2] = (res[0::2] - x.item()) / w.item()
    res[1::2] = (res[1::2] - y.item()) / h.item()
    # print("after", output)
    res = np.clip(res, 0, 1)
    return res


def crop_image(image, x1,y1,x2,y2):
    return image[:,x1:x2,y1:y2]

def my_loss(output, target):
    #mseloss (x - y) ^ 2
    #l1loss abs(x - y) 
#     output_2d = np.reshape(output, (-1, 2))
#     target_2d = np.reshape(target, (-1, 2))


    loss = torch.linalg.norm(output - target)
#     loss = torch.dist(output,target) # <---- result seemed quite good, and again with non cropped ones!

#     x = np.reshape(output.detach().cpu(), (-1, 2))
#     y = np.reshape(target.detach().cpu(), (-1, 2))
#     loss = torch.diag(torch.cdist(x,y)).mean()

#     loss = 0
#     for i in range(output.shape[0]):
#         for j in range(0, output.shape[1]):
#             loss += (torch.linalg.norm(output[i][j] - target[i][j]))
#     loss = loss / (output.shape[0] * output.shape[1])
    return loss
