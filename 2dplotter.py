

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import numpy as np
import helper
from model import PositionFinder, MotionDataset, BoundingBoxFinder
import pandas as pd

img_width = 256
test_transforms = transforms.Compose([
                            transforms.Resize((img_width,img_width)),
                            transforms.ToTensor()
                            ,transforms.Normalize([0.485, 0.456, 0.406], 
                                                [0.229, 0.224, 0.225])
                                      ])

maxval = 512
minval = 0

def positions():
    model = PositionFinder(img_width)
    model.load_state_dict(torch.load('model.m'))

    motions_test = MotionDataset('test2/input.csv', 'test2', test_transforms, (0,512), None)
    testloader = torch.utils.data.DataLoader(motions_test, batch_size=1, shuffle=True)
    images, details, labels, confidences = next(iter(testloader))
    positions = []
    positions_expected = []
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model.forward(images, details)
        logps_denormalized = logps 
        labels_denormalized = labels 
    #     print(logps_denormalized[0] * 10 - 5, labels_denormalized[0])
        for body_index in range(22):
            xyz = []
            xyz_e = []
            for pos_index in range(2):
                pos = logps_denormalized[0][body_index*2+pos_index]
                pos_e = labels_denormalized[0][body_index*2+pos_index]
                # pos = pos * (maxval - minval) + minval
                # pos_e = pos_e * (maxval - minval) + minval

                xyz.append(pos)
                xyz_e.append(pos_e)
            positions.append(xyz)
            positions_expected.append(xyz_e)

    positions = np.array(positions)
    positions_expected = np.array(positions_expected)

    xdata = positions.T[0]
    ydata = positions.T[1]

    xdata_e = positions_expected.T[0]
    ydata_e = positions_expected.T[1]
    print(images[0].shape)
    print(xdata)
    ax = helper.imshow(images[0], xdata=xdata, ydata=ydata)
    plt.show()

def boundaries(testloader = None):
    model = BoundingBoxFinder(img_width)
    model.load_state_dict(torch.load('bmodel.m'))
    if not testloader:
        motions_test = MotionDataset('test/input.csv', 'test', test_transforms, (0,512), None)
        testloader = torch.utils.data.DataLoader(motions_test, batch_size=1, shuffle=True)
    images, details, labels, confidences, boundaries = next(iter(testloader))
    positions = []
    positions_expected = []
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model.forward(images)
        x = logps[0][0]
        y = logps[0][1]
        x2 = logps[0][2]
        y2 = logps[0][3]
        xyz = [[x,y], [x2, y], [x, y2], [x2, y2]]
        xyz_e = [[confidences[0][0], confidences[0][2]], [confidences[0][1], confidences[0][3]]]
        positions.append(xyz)
        positions_expected.append(xyz_e)

    positions = np.array(positions)
    positions_expected = np.array(positions_expected)

    xdata = positions.T[0]
    ydata = positions.T[1]

    xdata_e = positions_expected.T[0]
    ydata_e = positions_expected.T[1]

    ax = helper.imshow(images[0], xdata=xdata, ydata=ydata)
    plt.show()

boundaries()
                                    







