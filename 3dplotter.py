from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import numpy as np
import helper
from model import Network, MotionDataset
import pandas as pd

#get min max
train_csv = pd.read_csv('train/input.csv')
test_csv = pd.read_csv('test/input.csv')
maxval = max(train_csv.iloc[:, 3:].to_numpy().max(), test_csv.iloc[:, 3:].to_numpy().max())
minval = min(train_csv.iloc[:, 3:].to_numpy().min(), test_csv.iloc[:, 3:].to_numpy().min())
print(minval, maxval)

# get min max of the depths
helper_arr = (np.arange(train_csv.shape[1]) + 1) % 3 == 0
helper_arr[0:5] = False
maxval_z = max(train_csv.iloc[:, helper_arr].to_numpy().max(), test_csv.iloc[:, helper_arr].to_numpy().max())
minval_z = min(train_csv.iloc[:, helper_arr].to_numpy().min(), test_csv.iloc[:, helper_arr].to_numpy().min())
print(minval_z, maxval_z)

img_width = 256
test_transforms = transforms.Compose([
                                       transforms.Resize((img_width,img_width)),
                                       transforms.ToTensor()
                                        ,transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])
                                      ])

model = Network(img_width)
model.load_state_dict(torch.load('model.m'))
# model.load_state_dict(torch.load('model_c4_l1sum.m'))
motions_test = MotionDataset('test/input.csv', 'test', test_transforms, (minval,maxval), (minval_z, maxval_z))
testloader = torch.utils.data.DataLoader(motions_test, batch_size=6, shuffle=True)
images, details, labels = next(iter(testloader))

helper.imshow(images[0], normalize=False)
print(images[0].min(), images[0].max())

# img = images[0].view(1, 195075)
# img = torch.zeros(1,120000) + 222
# Turn off gradients to speed up this part
positions = []
positions_expected = []
with torch.no_grad():
    print(images.shape)
    logps = model.forward(images, details)
    logps_denormalized = logps 
    labels_denormalized = labels 
    print(np.min(logps_denormalized[0].numpy()), np.min(labels_denormalized[0].numpy()))
    print(np.max(logps_denormalized[0].numpy()), np.max(labels_denormalized[0].numpy()))
#     print(logps_denormalized[0] * 10 - 5, labels_denormalized[0])
    for body_index in range(22):
        xyz = []
        xyz_e = []
        for pos_index in range(3):
            pos = logps_denormalized[0][body_index*3+pos_index]
            pos_e = labels_denormalized[0][body_index*3+pos_index]
            pos = pos * (maxval - minval) + minval
            pos_e = pos_e * (maxval - minval) + minval

            xyz.append(pos)
            xyz_e.append(pos_e)
        positions.append(xyz)
        positions_expected.append(xyz_e)
        print("exp", xyz_e)
        print(xyz)
#     print(list(model.parameters()))

positions = np.array(positions)
positions_expected = np.array(positions_expected)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

zdata = positions.T[1]
xdata = positions.T[0]
ydata = positions.T[2]
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
bigMarks = [4,5,9,12,17,20]
scale = [i if i not in bigMarks else 70 for i in range(len(xdata))]
ax.scatter3D(xdata, ydata, zdata, s=scale)
for i, pos in enumerate(positions):
    label = None
    if i == 4:
        label = "Neck"
    elif i == 5:
        label = "Head"
    elif i == 9:
        label = "LeftHand"
    elif i == 12:
        label = "LeftFoot"
    elif i == 17:
        label = "RightHand"
    elif i == 20:
        label = "RightFoot"
    if label is not None:
        ax.text(pos[0], pos[2], pos[1], label, None, color="green")

zdata_e = positions_expected.T[1]
xdata_e = positions_expected.T[0]
ydata_e = positions_expected.T[2]
ax.scatter3D(xdata_e, ydata_e, zdata_e, s=scale)


for i, pos in enumerate(positions_expected):
    label = None
    if i == 4:
        label = "Neck"
    elif i == 5:
        label = "Head"
    elif i == 9:
        label = "LeftHand"
    elif i == 12:
        label = "LeftFoot"
    elif i == 17:
        label = "RightHand"
    elif i == 20:
        label = "RightFoot"
    if label is not None:
        ax.text(pos[0], pos[2], pos[1], label, None)


plt.xlim(0, 512)
plt.ylim(0, 512)
ax.set_zlim(0,512)
# rotate the axes and update
for i in range(0, 2):
    for angle in range(0, 360, 90):
        ax.view_init(0, angle)
        plt.draw()
        plt.pause(4)
