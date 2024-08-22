import numpy as np
import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l
import random

def synthetic_data(w, b, num_examples):  # num_examples:n个样本
    '''生成 y=Xw+b+噪声'''
    X = torch.normal(0, 1, (num_examples, len(w)))  # 生成 X，他是一个均值为0，方差为1的随机数，他的大小: 行为num_examples，列为w的长度表示多少个feature
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # 加入一些噪音，均值为0 ，方差为0.01，形状和y是一样
    return X, y.reshape((-1, 1))  # 把X和y作为一个列向量返回


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

def data_iter(batch_size, features, labels):  # data_iter函数接收批量大小、特征矩阵和标签向量作为输入
    num_examples = len(features)
    indices = list(range(num_examples))  # 生成每个样本的index，随机读取，没有特定顺序。range随机生成0 —（n-1）,然后转化成python的list
    random.shuffle(indices)  # 将下标全都打乱，打乱之后就可以随机的顺序去访问一个样本
    for i in range(0, num_examples, batch_size):  # 每次从0开始到num_examples，每次跳batch_size个大小
        batch_indices = torch.tensor(  # 把batch_size的index找出来，因为可能会超出我们的样本个数，所以最后如果没有拿满的话，会取出最小值，所以使用min
            indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


batch_size = 10

for X, y in data_iter(batch_size, features, labels):  # 调用data_iter这个函数返回iterator（迭代器），从中拿到X和y
    print(X, '\n', y)  # 给我一些样本标号每次随机的从里面选取一个样本返回出来参与计算
    break

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)  # w:size为2行1列,随机初始化成均值为0，方差为0.01的正态分布，requires=true是指需要计算梯度
b = torch.zeros(1, requires_grad=True)  # 对于偏差来说直接为0，1表示为一个标量，因为也需要进行更新所以为True

def linreg(X, w, b):
    """线性回归模型。"""
    return torch.matmul(X, w) + b    #矩阵乘以向量再加上偏差

def squared_loss(y_hat, y):         #y_hat是预测值，y是真实值
    """均方损失。"""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2      #按元素做减法，按元素做平方，再除以2  （这里没有做均方）

def sgd(params, lr, batch_size):  # 优化算法是sgd，他的输入是：params给定所有的参数,这个是一个list包含了w和b，lr是学习率，和batch_size大小
    """小批量随机梯度下降。"""
    with torch.no_grad():  # 这里更新的时候不需要参与梯度计算所以是no_grad
        for param in params:  # 对于参数中的每一个参数，可能是w可能是b
            param -= lr * param.grad / batch_size  # 参数减去learning rate乘以他的梯度（梯度会存在.grad中）。上面的损失函数中没有求均值，所以这里除以了batch_size求均值，因为乘法对于梯度是一个线性的关系，所以除以在上面损失函数那里定义和这里是一样的效果
            param.grad.zero_()  # 把梯度设置为0，因为pytorch不会自动的设置梯度为0，需要手动，下次计算梯度的时候就不会与这次相关了


lr = 0.03  # 首先指定一些超参数：学习率为0.03
num_epochs = 3  # epoch为3表示把整个数据扫3遍
net = linreg  # network为linreg前面定义的线性回归模型
loss = squared_loss  # loss为均方损失

for epoch in range(num_epochs):  # 训练的过程基本是两层for循环（loop）,第一次for循环是对数据扫一遍
    for X, y in data_iter(batch_size, features, labels):  # 对于每一次拿出一个批量大小的X和y
        l = loss(net(X, w, b), y)  # 把X,w,b放进network中进行预测，把预测的y和真实的y来做损失，则损失就是一个长为批量大小的一个向量，是X和y的小批量损失
        # l(loss)的形状是（'batch_size',1）,而不是一个标量
        l.sum().backward()  # 对loss求和然后算梯度。计算关于['w','b']的梯度
        sgd([w, b], lr, batch_size)  # 算完梯度之后就可以访问梯度了，使用sgd对w和b进行更新。使用参数的梯度对参数进行更新
    # 对数据扫完一遍之后来评价一下进度，这块是不需要计算梯度的，所以放在no_grad里面
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)  # 把整个features，整个数据传进去计算他的预测和真实的labels做一下损失，然后print
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')

