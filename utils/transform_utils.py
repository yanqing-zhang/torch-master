'''
@Project ：torch-master 
@File    ：transform_utils.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/7/18 18:57 
'''
from torchvision import transforms
class TransformUtils:
    @staticmethod
    def get_train_transforms():
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010))])
        return train_transforms

    @staticmethod
    def get_test_transforms():
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010))])
        return test_transforms