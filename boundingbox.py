import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
from skimage import io, transform
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch import optim, nn

import helper
from model import MotionDataset, PositionFinder, BoundingBoxFinder

minmax, minmax_z = helper.get_minmax('train/input.csv', 'test/input.csv')

img_width = 256
train_transforms = transforms.Compose([
                                        transforms.Resize((img_width,img_width)),
                                       transforms.ToTensor()
                                       ,transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])
                                      ])

test_transforms = transforms.Compose([
                                       transforms.Resize((img_width,img_width)),
                                       transforms.ToTensor()
                                           ,transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])
                                      ])

motions = MotionDataset('train/input.csv', 'train', train_transforms, minmax, minmax_z)
trainloader = DataLoader(motions, batch_size=8, shuffle=True)
inp, output = next(iter(trainloader))

motions_test = MotionDataset('test/input.csv', 'test', test_transforms, minmax, minmax_z)
testloader = torch.utils.data.DataLoader(motions_test, batch_size=8, shuffle=True)


criterion = nn.L1Loss(reduction='sum')


def train(param):
    print(param)
    bmodel = BoundingBoxFinder(img_width, param)
    # # Optimizers require the parameters to optimize and a learning rate
    optimizer = optim.SGD(bmodel.parameters(), lr=param["learning"])
    device = 'cuda'
    bmodel.to(device)
    train_losses, test_losses = [], []
    epochs = param["epochs"]

    for e in range(epochs):
        running_loss = 0
        for inp, output in trainloader:
            for k in inp:
                inp[k] = inp[k].to(device)
            for k in output:
                output[k] = output[k].to(device)
            #1. forward pass
            ps = bmodel.forward(inp["images"])
            #2. calculate loss
            loss = criterion(ps, output["boundaries"])
            #0. Clear the gradients, do this because gradients are accumulated
            optimizer.zero_grad()

            #3. run backward propagation
            loss.backward()

            # 4. Take an update step and few the new weights
            optimizer.step()
            running_loss += loss.item()
    #         print(loss.item())
        else:
            test_loss = 0
            accuracy = 0
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                bmodel.eval()
                for inp, output in testloader:
                    for k in inp:
                        inp[k] = inp[k].to(device)
                    for k in output:
                        output[k] = output[k].to(device)
                    ps = bmodel.forward(inp["images"])
                    test_loss += criterion(ps, output["boundaries"])

            
            bmodel.train()
            
            test_loss = test_loss/len(testloader)
            train_loss = running_loss/len(trainloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"(Epoch {e+1}/{epochs}) Training loss: {train_loss}, Test loss: {test_loss}")

    PATH = 'bmodel.m'
    torch.save(bmodel.state_dict(), PATH)

params = {
    'epochs' : [4],
    'learning' : [0.004],
    'conv1_c' : [8],
    'conv1_bn' : [False],
    'conv2_c' : [16],
    'conv3_c' : [32],
    'drop' : [0.4],
    'hidden' : [128]
} 

for learning in params["learning"]:
    for conv1_c in params["conv1_c"]:
        for conv1_bn in params["conv1_bn"]:
            for conv2_c in params["conv2_c"]:
                for conv3_c in params["conv3_c"]:
                    for drop in params["drop"]:
                        for hidden in params["hidden"]:
                            for epochs in params["epochs"]:
                                param = {
                                    'epochs' : epochs,
                                    'learning' : learning,
                                    'conv1_c' :conv1_c,
                                    'conv1_bn' : conv1_bn,
                                    'conv2_c' : conv2_c,
                                    'conv3_c' : conv3_c,
                                    'drop' : drop,
                                    'hidden' : hidden
                                }
                                train(param)



