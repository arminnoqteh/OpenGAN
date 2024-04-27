import torch
import torch.utils
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import FashionMNIST
import os
import pickle
from tqdm import tqdm
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(7*7*64, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        return x



# Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")

# Instantiate the model
model = SimpleCNN().to(device)

# Load Fashion-MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_data = FashionMNIST(root='./data', train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False)

# Extract features and labels
features = []
labels = []
with torch.no_grad():
    for inputs, label in tqdm(train_data, desc='Extracting features from train data'):
        inputs = inputs.to(device)
        output = model(inputs)
        features.append(output.cpu())
        labels.append(label)

# Concatenate all feature vectors and labels
features = torch.cat(features, 0)
labels = torch.cat(labels, 0)

# Save features and labels as a pickle file
with open('./feats/res18_fmnist.pkl', 'wb') as f:
    pickle.dump({'WholeFeatVec': features, 'labels': labels}, f)