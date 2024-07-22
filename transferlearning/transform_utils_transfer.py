'''
@Project ：torch-master 
@File    ：transform_utils_transfer.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/7/22 16:01 
'''
from torchvision import transforms

class TransformUtils:
    @staticmethod
    def get_train_transform()->transforms.Compose:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return train_transform

    @staticmethod
    def get_test_transform():
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return test_transform