import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height, and width, but keep channel dimension
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    # std = sqrt(E[X^2] - (E[X])^2)
    std = torch.sqrt(channels_squared_sum / num_batches - mean**2)

    return mean, std

# Example Usage:
# 1. Define your dataset and transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

# Replace 'your_dataset_path' with the actual path to your image dataset
# For example, if using ImageFolder for a custom dataset:
dataset = datasets.ImageFolder('/media/NVME/PRIYO-DATA/EXPLAINABLE_DATASET/SGAN2_CELBAHQ256/train/2024/', transform=transform)
# Or, for a built-in dataset like CIFAR10:
# dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# 2. Create a DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# 3. Calculate mean and std
mean, std = get_mean_and_std(dataloader)

print(f"Calculated Mean: {mean}")
print(f"Calculated Standard Deviation: {std}")