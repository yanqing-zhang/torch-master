'''
@Project ：torch-master 
@File    ：deeplearn_cifar10_model.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/7/18 18:38 
'''
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from matplotlib import pyplot as plt
from utils.transform_utils import TransformUtils
from torch import nn, optim
from deeplearning.lenet_model import LeNet5, le_model, device
import torch

class DeepNet:

    def dataloader(self):
        '''
        加载CIFAR10数据
        :return:
        '''
        train_data = CIFAR10(root='./data/train/', train=True, download=True, transform=TransformUtils.get_train_transforms())
        self.show_img(train_data)
        test_data = CIFAR10(root='./data/test/', train=False, download=True, transform=TransformUtils.get_test_transforms())
        train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_data, batch_size=100, shuffle=False, num_workers=2)
        return train_loader, test_loader

    def show_img(self,raw_data):
        '''
        选其中第一个数据可视化
        :param raw_data:
        :return:
        '''
        data, label = raw_data[0]
        plt.imshow(data.permute(1,2,0))

    def get_model(self):
        '''
        获取模型
        :return:
        '''
        return le_model

    def fit(self):
        '''
        训练
        :return:
        '''
        losses = nn.CrossEntropyLoss()
        model = self.get_model()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        epochs = 10
        train_loader, test_loader = self.dataloader()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = losses(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss = loss.item()
            print("Epoch:{} Loss:{}".format(epoch, epoch_loss / len(train_loader)))

    def test(self):
        '''
        验证
        :return:
        '''
        num_correct = 0
        train_loader, test_loader = self.dataloader()
        model = self.get_model()
        for batch_index, data_test, target_test in enumerate(test_loader):
            model.eval()
            target_test = target_test.to(device)
            data_test = data_test.to(device)
            y_pred = model(data_test)
            _, y_predicted = torch.max(y_pred, 1)
            num_correct += (y_predicted == target_test).float().sum()
        accuracy = num_correct / (len(test_loader)*test_loader.batch_size)
        print(len(test_loader), test_loader.batch_size)
        print("Test Accuracy: {:.4f}".format(accuracy))

    def save_model(self):
        '''
        保存模型
        :return:
        '''
        model = self.get_model()
        torch.save(model.state_dict(), './data/models/model.pth')

    def load_model(self):
        '''
        加载模型
        :return:
        '''
        model = self.get_model()
        model.load_state_dict(torch.load('./data/models/model.pth'))

if __name__ == '__main__':
    model = DeepNet()
    model.fit()