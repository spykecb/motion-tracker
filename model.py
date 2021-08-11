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
        boundaries = boundaries / self.img_size

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
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2_s = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.conv2_1 = nn.Conv2d(64, 64, 1, padding=0)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3_s = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.conv3_1 = nn.Conv2d(128, 128, 1, padding=0)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_s = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.conv4_1 = nn.Conv2d(256, 256, 1, padding=0)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.avgpool = nn.AvgPool2d(2,2)
        # self.roi_size = 16
        # self.roi = ops.PSRoIAlign((self.roi_size,self.roi_size), 1/16, 2)

        size = self.img_size//16 * self.img_size//16
        #Linear layers
        self.hidden1 = nn.Linear(256*size + 2, 500)
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
        self.output.weight.data.zero_()
        self.output.bias.data.zero_()
        
    def forward(self, x, details, bboxes):
        #Convolutional layers
        out = self.conv1_bn(F.relu(self.conv1(x)))
        out = F.relu(self.conv2_s(out))
        res1 = out
        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2(out))
        out += res1

        out = F.relu(self.conv3_s(out))
        res2 = out
        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3(out))
        out += res2

        out = F.relu(self.conv4_s(out))
        res3 = out
        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4(out))
        out += res3
        out = self.avgpool(out)

        # make sure input tensor is flattened
        # torch.Size([16, 512, 16, 16])
        size = self.img_size//16 * self.img_size//16
        out = out.view(-1, 256*size)
        out = torch.cat((out,details), dim = 1)
        # x = torch.cat((out,details, bboxes), dim = 1)
        

        #Forward pass through the network, returns the output
        out = self.dropout(F.relu(self.hidden1(out)))
        # x = self.dropout(F.relu(self.hidden2(out)))
        out = self.output(out)

        #confidences
        # y = x[:,5*2:5*3]
        # x = x[:,:5*2]
        out = out.view(out.shape[0], -1, 2)
        # y = torch.sigmoid(y)
        # y = F.log_softmax(y, dim=1)
        return out

class BoundingBoxFinder(nn.Module):
    def __init__(self, img_size, param = None):
        super().__init__()
        self.img_size = img_size
        if param is None:
            param = {
                'epochs' : 8,
                'learning' : 0.004,
                'conv1_c' : 8,
                'conv1_bn' : False,
                'conv2_c' : 16,
                'conv3_c' : 32,
                'drop' : 0.4,
                'hidden' : 128
            } 
        self.param = param
        #defining the layers here

        #Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2_s = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.conv2_1 = nn.Conv2d(64, 64, 1, padding=0)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3_s = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.conv3_1 = nn.Conv2d(128, 128, 1, padding=0)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4_s = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.conv4_1 = nn.Conv2d(256, 256, 1, padding=0)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.avgpool = nn.AvgPool2d(2,2)

        size = self.img_size//16 * self.img_size//16
        #Linear layers
        self.hidden1 = nn.Linear(256*size, 1000)
        # self.hidden2 = nn.Linear(1000, 500)
        self.output = nn.Linear(1000, 4)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=param["drop"])

    def initialize(self):
        # for conv in [self.conv1, self.conv1_bn, self.conv1_s, 
        #             self.conv2, self.conv2_s,
        #             self.conv3, self.conv3_s,
        #             self.conv4 ]:
        #     conv.weight.data.zero_()
        #     conv.bias.data.zero_()
        self.hidden1.weight.data.zero_()
        self.hidden1.bias.data.zero_()
        self.output.weight.data.zero_()
        self.output.bias.data.zero_()
        
    def forward(self, x):
        #Convolutional layers
        out = self.conv1(x)
        if self.param["conv1_bn"]:
            out = self.conv1_bn(out)
        out = F.relu(out)
        
        out = self.conv2_s(out)
        out = F.relu(out)

        res1 = out
        out = self.conv2_1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out += res1

        out = self.conv3_s(out)
        out = F.relu(out)
        res2 = out
        out = self.conv3_1(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = F.relu(out)
        out += res2

        out = self.conv4_s(out)
        out = F.relu(out)
        res3 = out
        out = self.conv4_1(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = F.relu(out)
        out += res3
        out = self.avgpool(out)

        # make sure input tensor is flattened
        size = self.img_size//16 * self.img_size//16
        out = out.view(-1, 256*size)

        #Forward pass through the network, returns the output
        out = self.dropout(F.relu(self.hidden1(out)))
        out = self.output(out)
        
        return out

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
