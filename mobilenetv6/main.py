import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import os

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

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create an instance of the MobileNetV2 model
model = CNN().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
checkpoint_dir = 'model'
os.makedirs(checkpoint_dir, exist_ok=True)
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Evaluate the model on the test set
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    
    # Print epoch results
    accuracy = 100 * total_correct / total_samples
    print(f"Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy:.2f}%")

    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved at '{checkpoint_path}'")