'''
@Project ：torch-master 
@File    ：dataset_example.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/7/25 17:32 
'''
import torch
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    """
    目标：通过原始图像image和对应的标注label使用代码包装成pytorch的dataset数据集
    数据来源:https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    数据介绍:数据包是关于花朵分类的数据
    ./datas/flowers:
    ├─daisy
    ├─dandelion
    ├─roses
    ├─sunflowers
    └─tulips
    """
    def __init__(self, images_path: list, labels_path: list, transform=None):
        super(MyDataset, self).__init__()
        self.images_path = images_path
        self.labels_path = labels_path
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        image = Image.open(self.images_path[index])
        if image.mode != 'RGB':
            raise ValueError(f"image:{self.images_path[index]} is not RGB mode")
        label = self.labels_path[index]
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    @staticmethod
    def collate_fn(batch):
        '''
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        zip() 函数用于将多个可迭代对象（如列表、元组等）聚合在一起，
        并返回一个元组迭代器，其中每个元组包含来自每个可迭代对象的元素
        *batch 是一个解包操作，它将batch中的每个元组解包成单独的参数传递给 zip()

        例如，如果 batch 是 [(img1, label1), (img2, label2), (img3, label3)]，
        那么 zip(*batch) 将返回一个迭代器，它在第一次迭代时返回 (img1, img2, img3)，
        在第二次迭代时返回 (label1, label2, label3)。

        tuple(zip(*batch)): 将 zip(*batch) 返回的迭代器转换成一个元组。在上面的例子中，
        这将产生一个包含两个元组的元组：((img1, img2, img3), (label1, label2, label3))。
        :return:
        '''
        print(f"batch:{len(batch)}")
        images, labels = tuple(zip(*batch))
        print(f"images:{images}")
        print(f"labels:{labels}")
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

