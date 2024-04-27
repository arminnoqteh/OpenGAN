import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import pickle
from tqdm import tqdm

# Set device
device = torch.device("mps")

# Load pre-trained model
model = models.resnet18(pretrained=True)
model = model.to(device)
model.eval()

# Remove the last layer (classification layer)
model = torch.nn.Sequential(*(list(model.children())[:-1]))

# Define transformations (you can modify this part based on your needs)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load Tiny ImageNet dataset
data_dir = '/Users/sana/Downloads/tiny-imagenet-200/train'
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Create data loader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

# Extract features and labels
features = []
labels = []
with torch.no_grad():
    for inputs, label in tqdm(data_loader, desc="Feature extraction"):
        inputs = inputs.to(device)
        output = model(inputs)
        features.append(output.cpu())
        labels.append(label)

# Concatenate all feature vectors and labels
features = torch.cat(features, 0)
labels = torch.cat(labels, 0)

# Save features and labels as a pickle file
with open('./feats/res18.pkl', 'wb') as f:
    pickle.dump({'WholeFeatVec': features, 'labels': labels}, f)