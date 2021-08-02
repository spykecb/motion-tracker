import torch
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        #defining the layers here

        #Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)

        size = self.img_size//8 * self.img_size//8
        #Linear layers
        self.hidden1 = nn.Linear(64*size, 500)
        # self.hidden2 = nn.Linear(n_hidden_units, n_hidden_units//2)
        self.output = nn.Linear(500, 1)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        #Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # make sure input tensor is flattened
        size = self.img_size//8 * self.img_size//8
        x = x.view(-1, 64*size)

        #Forward pass through the network, returns the output
        x = self.dropout(F.relu(self.hidden1(x)))
        # x = self.dropout(F.relu(self.hidden2(x)))
        x = torch.sigmoid(self.output(x)).squeeze(1)
        return x