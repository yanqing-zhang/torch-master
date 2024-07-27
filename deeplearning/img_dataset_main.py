'''
@Project ：torch-master 
@File    ：img_dataset_main.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/7/27 17:57 
'''

import os
import torch
from torchvision import transforms
from deeplearning.utils.data_util import DataUtils
from deeplearning.dataset_example import MyDataset
class ImgDatasetTestCase:

    def data_handle(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))  # 当前项目路径
        print(f"cur_path: {cur_path}")
        data_path = os.path.join(os.path.dirname(cur_path), 'datas')  # datas
        print(f"data_path: {data_path}")
        root = data_path + "/flowers"
        print(f"root:{root}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_images_path, train_images_label, val_images_path, val_images_label = DataUtils.read_split_data(root)
        train_data_set = MyDataset(images_path=train_images_path,
                                   labels_path=train_images_label,
                                   transform=DataUtils.transform()["train"])
        batch_size = 8
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers'.format(nw))
        train_loader = torch.utils.data.DataLoader(train_data_set,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=nw,
                                                   collate_fn=train_data_set.collate_fn)
        # plot_data_loader_image(train_loader)

        for step, data in enumerate(train_loader):
            images, labels = data
            print(f'step: {step}, images: {images}, labels: {labels}')

if __name__ == '__main__':
    ImgDatasetTestCase().data_handle()