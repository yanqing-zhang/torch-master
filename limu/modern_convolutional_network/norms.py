'''
@Project ：torch-master 
@File    ：norms.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/3/7 9:43 
'''
import torch
import torch.nn as nn
from d2l.torch_util import load_data_fashion_mnist, train_ch6, try_gpu

def batch_norm(x, gamma, beta, moving_mean, moving_var, eps, momentum):
    """
    根据数学公式翻译成代码
    """
    if not torch.is_grad_enabled():
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(x.shape) in (2, 4)
        if len(x.shape) == 2:
            mean = x.mean(dim=0)
            var= ((x - mean)**2).mean(dim=0)
        else:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = ((x - mean)**2).mean(dim=(0, 2, 3), keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * x_hat + beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    """
    自定义BatchNorm层
    """
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)

        Y, self.moving_mean, self.moving_var = batch_norm(x, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return Y

def build_lenet_with_batch_norm():
    """
    使用batch norm对LeNet模型进行重构
    """
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5),
        BatchNorm(6, num_dims=4),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5),
        BatchNorm(16, num_dims=4),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16*4*4, 120),
        BatchNorm(120, num_dims=2),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        BatchNorm(84, num_dims=2),
        nn.Sigmoid(),
        nn.Linear(84, 10)
    )

def fit():
    """
    使用自定义的LeNet进行模型训练
    """
    net = build_lenet_with_batch_norm()
    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())