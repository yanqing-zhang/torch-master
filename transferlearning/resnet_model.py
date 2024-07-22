'''
@Project ：torch-master 
@File    ：resnet_model.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/7/22 16:33 
'''
from torchvision import models
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from transferlearning.datas_utils import DataUtils
import os
class ResNetModel:

    def get_model(self):
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        return model

    def fit(self):
        model = self.get_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),
                              lr=0.001,
                              momentum=0.9)
        exp_lr_scheduler = StepLR(optimizer,
                                  step_size=7,
                                  gamma=0.1)
        epochs = 25
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0
            train_loader, test_loader = DataUtils.data_loaders()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()/inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)/inputs.size(0)
                exp_lr_scheduler.step()
                train_epoch_loss = running_loss/len(train_loader)
                train_epoch_acc = running_corrects/len(train_loader)

                model.eval()
                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()/inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)/inputs.size(0)
                    epoch_loss = running_loss/len(test_loader)
                    epoch_acc = running_corrects.double()/len(test_loader)
                    print("Train: Loss: {:.4f} Acc: {:.4f} Val: Loss: {:.4f} Acc: {:.4f}".format(train_epoch_loss,
                                                                                                 train_epoch_acc,
                                                                                                 epoch_loss,
                                                                                                 epoch_acc))

if __name__ == '__main__':
    os.environ["http_proxy"] = "http://127.0.0.1:10792"
    os.environ["https_proxy"] = "http://127.0.0.1:10792"
    DataUtils.download_and_unzip_data()
    model = ResNetModel()
    model.fit()