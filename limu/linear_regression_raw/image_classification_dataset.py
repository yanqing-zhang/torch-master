'''
@Project ：torch-master 
@File    ：image_classification_dataset.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/8/23 17:47 
'''
import torch
import torchvision
from torch.utils import data
from torchvision import transforms

class ImageClassificationDataset:

    def get_fashion_mnist(self):
        transform = transforms.ToTensor()
        train_mnist_data = torchvision.datasets.FashionMNIST(
            root='../data', train=True, transform=transform, download=True)
        test_mnist_data = torchvision.datasets.FashionMNIST(
            root='../data', train=False, transform=transform, download=True)
        return train_mnist_data, test_mnist_data