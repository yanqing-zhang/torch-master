'''
@Project ：torch-master 
@File    ：data_util.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/7/27 9:38 
'''
import os
import json
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
class DataUtils:

    def read_split_data(root:str, val_rate: float = 0.2):
        random.seed(0)
        # 通过断言判断根目录是否存在
        assert os.path.exists(root), "dataset root: {} does not exist!".format(root)
        # 遍历根目录，如果根目录下的文件和文件夹是文件夹的话，把该文件夹罗列成列表
        flower_category = [cate for cate in os.listdir(root) if os.path.isdir(os.path.join(root, cate))]
        # 对花类文件夹列表排序
        flower_category.sort()
        # 把 花名:序号 翻转成 序号:花名
        cate_indices = dict((key, val) for val, key in enumerate(flower_category))
        # 把序号:花名 字典转成json写入文件cate_indices.json
        json_str = json.dumps(dict((val, key) for key, val in cate_indices.items()), indent=4)
        with open('cate_indices.json', 'w') as f:
            f.write(json_str)
        train_images_path = []
        train_images_label = []
        val_images_path = []
        val_images_label = []
        every_cate_num = []
        supported = [".jpg","JPG",".jpeg", ".png", ".PNG"]
        for category in flower_category:
            cate_path = os.path.join(root, category)
            '''
            os.path.splitext(i): 这个函数将路径i分割成两部分：文件名和扩展名。返回一个元组，其中第一个元素是文件名，
            第二个元素是文件的扩展名（包括点号.）。
            例如，如果i是'example.txt'，那么os.path.splitext(i)将返回('example', '.txt')。
            os.path.splitext(i)[-1]: 通过[-1]索引，我们从os.path.splitext(i)返回的元组中提取出扩展名。
            在上面的例子中，这将得到'.txt'。
            '''
            # os.listdir(cate_path) 列出各花类文件夹下的花图片，并进行文件名与扩展名的分割处理
            # 判断扩展名要落在可支持的图片扩展名列表里
            # 然后这符合条件的图片按：根目录+花类文件夹+花图片，连接成路径串构成列表
            images = [os.path.join(root, category, i) for i in os.listdir(cate_path)
                      if os.path.splitext(i)[-1] in supported]
            # 获取花名编号并追加到every_cat_num列表里
            image_cate = cate_indices[category]
            every_cate_num.append(len(images))
            # 按20%在图片中取样作为验证图片数据的路径
            val_path = random.sample(images, k=int(val_rate*len(images)))
            # 遍历整个数据集图片，判断在上面取样列表中的作为验证集，否则为训练集
            for image_path in images:
                if image_path in val_path:
                    val_images_path.append(image_path)
                    val_images_label.append(image_cate)
                else:
                    train_images_path.append(image_path)
                    train_images_label.append(image_cate)
            print(f"{sum(every_cate_num)} images were found in the dataset.")
            print(f"{len(train_images_path)} images for training.")
            print(f"{len(val_images_path)} images for validation.")
            # 绘图显示每个分类花的图片数量
            plot_images = False
            if plot_images:
                plt.bar(range(len(flower_category)), every_cate_num, align='center')
                plt.xticks(range(len(flower_category)), flower_category)
                for i, v in enumerate(every_cate_num):
                    plt.text(x=i, y=v + 5, s=str(v), ha='center')
                plt.xlabel('image category')
                plt.ylabel('number of images')
                plt.title("flower category distribution")
                plt.show()
            return train_images_path, train_images_label, val_images_path, val_images_label

    def plot_data_loader_image(data_loader):
        batch_size = data_loader.batch_size
        plot_num = min(batch_size, 4)
        json_path = './cate_indices.json'
        assert os.path.exists(json_path), "cate_indices.json does not exist!"
        with open(json_path, 'r') as f:
            cate_indices = json.load(f)
        for data in data_loader:
            images, labels = data
            for i in range(plot_num):
                '''
                .transpose((1, 2, 0)): NumPy数组的.transpose()方法用于重新排列数组的维度。
                这里的(1, 2, 0)是一个元组，指定了新的维度顺序。假设原始图像的形状是(C, H, W)，
                其中C是通道数（通常是3，代表RGB），H是高度，W是宽度。.transpose((1, 2, 0))
                会将维度从(C, H, W)更改为(H, W, C)，即将通道维移动到最后，这是许多图像处理库（如OpenCV）期望的格式。
                '''
                image = images[i].numpy().transpose((1, 2, 0))
                '''
                这行代码用于对图像进行标准化和反标准化。在深度学习中，图像通常需要经过预处理，
                其中一步是将图像像素值从[0, 255]范围标准化到[0, 1]范围。[0.229, 0.224, 0.225]和[0.485, 0.456, 0.406]
                分别是图像每个通道的均值和标准差，这些值通常是从训练数据集中计算得到的。
                (image * [0.229, 0.224, 0.225]): 将图像每个通道乘以其相应的标准差。
                + [0.485, 0.456, 0.406]: 将上一步的结果加上每个通道的均值。
                最后再乘以255：将图像的像素值从[0, 1]范围转换回原始的[0, 255]范围。
                '''
                image = (image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
                '''
                .item(): 如果labels[i]是一个单元素张量（例如，只有一个类别的标签），
                .item()方法将提取这个张量中的Python标量值。
                '''
                label = labels[i].item()
                plt.subplot(1, plot_num, i + 1)
                plt.xlabel(cate_indices[str(label)])
                plt.xticks([])
                plt.yticks([])
                plt.imshow(image.astype(np.uint8))
            plt.show()

    def write_pickle(list_info: list, file_name: str):
        '''
        写文件
        :param file_name:
        :return:
        '''
        with open(file_name, 'wb') as f:
            pickle.dump(list_info, f)

    def read_pickle(file_name: str) -> list:
        '''
        读文件
        :return:
        '''
        with open(file_name, 'rb') as f:
            info_list = pickle.load(f)
            return info_list

    def transform():
        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
        return data_transform


if __name__ == '__main__':
    pass