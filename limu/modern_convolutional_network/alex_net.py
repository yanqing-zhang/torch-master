'''
@Project ：torch-master 
@File    ：alex_net.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/3/7 14:49 
'''
import torch
import torch.nn as nn
from d2l.torch_util import load_data_fashion_mnist, train_ch6, try_gpu
def build_alex_net_structure():
    net = nn.Squential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nn.Conv2d(256, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        nn.Linear(6400, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 10)
    )

def show_axlex_net_structure():
    net = build_alex_net_structure()
    x = torch.randn(1, 1, 224, 224)
    for layer in net:
        x = layer(x)
        print(f"{layer.__class__.__name__} output shape:\t {x.shape}")

def fit():
    net = build_alex_net_structure()
    batch_size = 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    lr, num_epochs = 0.01, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())