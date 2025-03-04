'''
@Project ：torch-master 
@File    ：googlenet_net.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/3/4 10:00 
'''
import torch
import torch.nn as nn
from inception_net import Inception
from d2l.torch_util import load_data_fashion_mnist, train_ch6, try_gpu
class GoogLeNet(nn.Module):
    def build_googlenet_structure(self):
        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                           nn.ReLU(),
                           nn.Conv2d(64, 192, kernel_size=3, padding=1),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                           Inception(256, 128, (128, 192), (32, 96), 64),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                           Inception(512, 160, (112, 224), (24, 64), 64),
                           Inception(512, 128, (128, 256), (24, 64), 64),
                           Inception(512, 112, (144, 288), (32, 64), 64),
                           Inception(528, 256, (160, 320), (32, 128), 128),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                           Inception(832, 384, (192, 384), (48, 128), 128),
                           nn.AdaptiveAvgPool2d((1, 1)),
                           nn.Flatten())

        net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
        return net

net = GoogLeNet().build_googlenet_structure()
def show_net_structure():
    x = torch.rand(size=(1, 1, 96, 96))
    for layer in net:
        x = layer(x)
        print(f"{layer.__class__.__name__} `s output shape:\t{x.shape}")

def fit():
    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter, = load_data_fashion_mnist(batch_size, resize=96)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device:{device}")
    print("==start training======================================")
    train_ch6(net, train_iter, test_iter, num_epochs, lr, device)

if __name__ == '__main__':
    show_net_structure()
    fit()

