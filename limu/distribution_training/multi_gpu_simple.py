'''
@Project ：torch-master 
@File    ：multi_gpu_simple.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/5/23 11:09 
'''
"""
# 加下面一行代码是为了解决这个问题：RuntimeError: module must have its parameters and buffers on device cuda:0(device_ids[0]) but found one of them on device: cpu
# 使用多 GPU 需要有一个主 GPU，来把每个 batch 的数据分发到每个 GPU，并从每个 GPU 收集计算好的结果。如果不指定主 GPU，那么数据就直接分发到每个 GPU，会造成有些数据在某个 GPU，而另一部分数据在其他 GPU，计算出错。
# 正在调试，目前看好像加这条代码也解决不了问题
device = torch.device("cuda:0")
"""
import torch
from torch import nn
from d2l.torch_util import load_data_fashion_mnist, try_gpu,evaluate_accuracy_gpu,try_all_gpus,sgd
from d2l.torch_util import Residual,Timer, Animator

def resnet18(num_classes, in_channels=1):
    """稍加修改的ResNet-18模型"""
    def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # 该模型使用了更小的卷积核、步长和填充，而且删除了最大汇聚层
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
    return net

net = resnet18(10)
# 获取GPU列表
devices = try_all_gpus()
# 我们将在训练代码实现中初始化网络

def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    devices = [try_gpu(i) for i in range(num_gpus)]
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)
    # 在多个GPU上设置模型
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = Timer(), 10
    animator = Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpu(net, test_iter),))
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'f'在{str(devices)}')

print(f"starting num_gpus=1 handle........")
train(net, num_gpus=1, batch_size=256, lr=0.1)
print(f"ended num_gpus=1 handle........")
print("======================================")
print(f"starting num_gpus=2 handle........")
train(net, num_gpus=2, batch_size=512, lr=0.2)
print(f"ended num_gpus=2 handle........")