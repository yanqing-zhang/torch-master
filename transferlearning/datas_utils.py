'''
@Project ：torch-master 
@File    ：datas_utils.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/7/22 16:02 
'''
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from torchvision import datasets, models
from transferlearning.transform_utils_transfer import TransformUtils
import torch

class DataUtils:

    @staticmethod
    def download_and_unzip_data():
        zip_url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'
        with urlopen(zip_url) as zip_file:
            with ZipFile(BytesIO(zip_file.read())) as zf:
                zf.extractall('./datas')

    @staticmethod
    def data_loaders():
        train_dataset = datasets.ImageFolder(
            root='./datas/hymenoptera_data/train',
            transform=TransformUtils.get_train_transform()
        )
        val_dataset = datasets.ImageFolder(
            root='./datas/hymenoptera_data/val',
            transform=TransformUtils.get_test_transform()
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=4
        )
        test_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=4
        )
        return train_loader, test_loader