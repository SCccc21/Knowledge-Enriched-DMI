import torchvision.transforms as transforms
from torchvision.datasets import CelebA
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10


# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Download and load the CelebA dataset

"""
Note:
    There are occasions when downloading the CelebA dataset using `torchvision`
    results in a Google Drive error. This is due to the dataset being hosted on
    Google Drive, which sometimes restricts the number of download requests.

    In such cases, an alternative is to download the CelebA dataset directly 
    from Kaggle using the following link:
    https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
"""

print("Starting download of CelebA dataset...")
dataset = CelebA(root='./data', split='train', transform=transform, download=True)
print("Completed download of CelebA dataset.\n")

# Download the MNIST dataset
print("Starting download of MNIST dataset...")
dataset = MNIST(root='./data', transform=transform, download=True)
print("Completed download of MNIST dataset.\n")

# Download the CIFAR10 dataset
print("Starting download of CIFAR10 dataset...")
dataset = CIFAR10(root='./data', transform=transform, download=True)
print("Completed download of CIFAR10 dataset.\n")