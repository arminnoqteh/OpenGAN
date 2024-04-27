import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import pickle
import numpy as np
from tqdm import tqdm

# Transformations applied on each image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to fit the ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizing for grayscale
])

# Load Fashion-MNIST dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load pre-trained ResNet-18
model = models.resnet18(pretrained=True)

# Adjusting the first layer to accept 1 channel input
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Removing the final fully connected layer to extract features before classification
feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))

# If you have a GPU, transfer the model to GPU
device = torch.device("mps")
feature_extractor = feature_extractor.to(device)

features = []
labels = []

feature_extractor.eval()  # Set the model to evaluation mode

with torch.no_grad():  # No need to compute gradient for feature extraction
    for inputs, batch_labels in tqdm(train_loader, desc='Extracting features from train data'):
        inputs = inputs.to(device)
        batch_features = feature_extractor(inputs)
        batch_features = batch_features.view(batch_features.size(0), -1)  # Flatten the features

        features.extend(batch_features.cpu().detach().numpy())
        labels.extend(batch_labels.numpy())

# Convert list to array for easier handling later
features = np.array(features)
labels = np.array(labels)

# Create directory if it does not exist
os.makedirs('./feats', exist_ok=True)

# Save as pickle file
with open('./feats/res18_fmnist.pkl', 'wb') as f:
    pickle.dump({'WholeFeatVec': features, 'labels': labels}, f)

