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
maxval = max(train_csv.drop([train_csv.columns[0], train_csv.columns[1]],axis=1).to_numpy().max(), test_csv.drop([test_csv.columns[0], test_csv.columns[1]],axis=1).to_numpy().max())
minval = min(train_csv.drop([train_csv.columns[0], train_csv.columns[1]],axis=1).to_numpy().min(), test_csv.drop([test_csv.columns[0], test_csv.columns[1]],axis=1).to_numpy().min())

img_width = 256
test_transforms = transforms.Compose([
                                       transforms.Resize((img_width,img_width)),
                                       transforms.ToTensor()
                                        ,transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])
                                      ])

model = Network(img_width)
model.load_state_dict(torch.load('model.m'))
motions_test = MotionDataset('test/input.csv', 'test', test_transforms, (minval,maxval))
testloader = torch.utils.data.DataLoader(motions_test, batch_size=6, shuffle=True)
images, heights, labels = next(iter(testloader))

helper.imshow(images[0], normalize=False)

# img = images[0].view(1, 195075)
# img = torch.zeros(1,120000) + 222
# Turn off gradients to speed up this part
print(labels.shape)
positions = []
positions_expected = []
with torch.no_grad():
    logps = model.forward(images, heights)
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
ax.scatter3D(xdata, ydata, zdata)
for i, pos in enumerate(positions):
    label = None
    if i == 4:
        label = "Neck"
    elif i == 5:
        label = "Head"
    elif i == 7:
        label = "LeftArm"
    elif i == 8:
        label = "LeftForeArm"
    elif i == 9:
        label = "LeftHand"
    elif i == 15:
        label = "RightArm"
    elif i == 16:
        label = "RightForeArm"
    elif i == 17:
        label = "RightHand"
    if label is not None:
        ax.text(pos[0], pos[2], pos[1], label, None)

zdata_e = positions_expected.T[1]
xdata_e = positions_expected.T[0]
ydata_e = positions_expected.T[2]
ax.scatter3D(xdata_e, ydata_e, zdata_e, marker='^')


# for i, pos in enumerate(positions_expected):
#     label = str(i)
#     if i == 4:
#         label = "Neck"
#     elif i == 5:
#         label = "Head"
#     elif i == 7:
#         label = "LeftArm"
#     elif i == 8:
#         label = "LeftForeArm"
#     elif i == 9:
#         label = "LeftHand"
#     elif i == 15:
#         label = "RightArm"
#     elif i == 16:
#         label = "RightForeArm"
#     elif i == 17:
#         label = "RightHand"
#     if label is not None:
#         ax.text(pos[0], pos[2], pos[1], label, None)

plt.xlim(0, 512)
plt.ylim(0, 10)
ax.set_zlim(0,512)
# rotate the axes and update
for i in range(0, 2):
    for angle in range(0, 360):
        ax.view_init(0, angle)
        # ax.view_init(0, -90)
        plt.draw()
        plt.pause(.001)
