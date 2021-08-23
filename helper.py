import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import torch
import torchvision
from torch import nn, optim, Tensor
from torch.autograd import Variable
from typing import Optional, List


# first 44 prime numbers
prime_numbers = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193]

def encode(x):
    res = 0
    for i in range(len(x)):
        a = x[i]
        p = prime_numbers[i]
        res += p ** a 
        print(res)
    return res        

def imshow(image, ax=None, title=None, normalize=True, xdata=[], ydata=[], xdata_e=[], ydata_e=[], bboxes=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.scatter(xdata * 256, [y*256 for y in ydata], color='red', s=100)
    ax.scatter(xdata_e * 256, [y*256 for y in ydata_e], color='green', s=50)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


def get_minmax_quaternion(train_path, test_path):
    #get min max
    train_csv = pd.read_csv(train_path)
    test_csv = pd.read_csv(test_path)
    maxval = max(train_csv.iloc[:, 3:].to_numpy().max(), test_csv.iloc[:, 3:].to_numpy().max())
    minval = min(train_csv.iloc[:, 3:].to_numpy().min(), test_csv.iloc[:, 3:].to_numpy().min())

    return (minval, maxval)

def get_minmax(train_path, test_path):
    #get min max
    train_csv = pd.read_csv(train_path)
    test_csv = pd.read_csv(test_path)
    until = 3 + 22 * 2
    maxval = max(train_csv.iloc[:, 3:until].to_numpy().max(), test_csv.iloc[:, 3:until].to_numpy().max())
    minval = min(train_csv.iloc[:, 3:until].to_numpy().min(), test_csv.iloc[:, 3:until].to_numpy().min())

    # get min max of the depths
    helper_arr = (np.arange(train_csv.shape[1]) + 1) % 3 == 0
    helper_arr[0:5] = False
    maxval_z = max(train_csv.iloc[:, helper_arr].to_numpy().max(), test_csv.iloc[:, helper_arr].to_numpy().max())
    minval_z = min(train_csv.iloc[:, helper_arr].to_numpy().min(), test_csv.iloc[:, helper_arr].to_numpy().min())

    return (minval, maxval), (minval_z, maxval_z)

def evaluate_model(model, inp, expected):
    positions = []
    positions_expected = []
    # Turn off gradients to speed up this part
    with torch.no_grad():
        for k in inp:
            inp[k] = inp[k].to('cpu')
        for k in expected:
            expected[k] = expected[k].to('cpu')
        
        logps = model.forward(inp["images"], inp["details"], inp["bboxes"])
        logps_denormalized = logps 
        labels_denormalized = expected["positions"]
        for body_index in range(22):
            xyz = []
            xyz_e = []
            for pos_index in range(2):
                xyz.append(logps_denormalized[0][body_index][pos_index])
                xyz_e.append(labels_denormalized[0][body_index][pos_index])
            positions.append(xyz)
            positions_expected.append(xyz_e)
    #     print(list(model.parameters()))

    positions = np.array(positions)
    positions_expected = np.array(positions_expected)
    return positions, positions_expected

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def get_label(i):
    label = ""
    # if i == 0:
    #     label = "Head"
    # elif i == 1:
    #     label = "LeftHand"
    # elif i == 2:
    #     label = "RightHand"
    # elif i == 3:
    #     label = "LeftFoot"
    # elif i == 4:
    #     label = "RightFoot"

    if i == 5:
        label = "Head"
    elif i == 9:
        label = "LeftHand"
    elif i == 17:
        label = "RightHand"
    elif i == 12:
        label = "LeftFoot"
    elif i == 20:
        label = "RightFoot"

    return label

