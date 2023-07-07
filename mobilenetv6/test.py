import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        self.linear = nn.Linear(16*49, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x) - self.relu(x-1)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu(x) - self.relu(x-1)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


# Load MNIST dataset
train_dataset = MNIST(root='data/', train=True, transform=ToTensor(), download=True)
test_dataset = MNIST(root='data/', train=False, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = CNN()
model.load_state_dict(torch.load("model/model_epoch6.pth"))
model.eval()

# relu6_layer1 = model.features[-7]  # Second-to-last ReLU6 layer
# relu6_layer2 = model.features[-3]  

with torch.no_grad():
    for images, _ in test_loader:
        x = images
        
        output1 = None
        output2 = None

        # Forward pass through the model
        x = model.conv1(x)
        x = model.relu(x) - model.relu(x-1)
        x = model.pool1(x)
        
        x = model.conv2(x)
        output1 = model.relu(x) - model.relu(x-1)



        channel_index = 0  # Index of the channel to visualize

        # Get the channel from output1 and output2
        channel_output1 = output1[0, :, :, :]

        # Visualize the channel outputs
        channel_output1 = channel_output1.reshape((14*4, 14*4))
        count1 = np.count_nonzero(channel_output1 == 1)

        if count1 >= 14*4*9:

            plt.figure(figsize=(100, 100))
            # plt.subplot(1, 2, 1)
            plt.imshow(channel_output1.cpu(), cmap='gray')
            # plt.title("Output1 - Channel {}".format(channel_index))
            plt.axis('off')
            
            plt.savefig('figs/CNN.jpg')
            count1 = 0

