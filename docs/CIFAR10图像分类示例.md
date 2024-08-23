# CIFAR10图像分类示例全流程

## 0、说明

本示例主要演示图像分类的全流程，从数据加载、变换、数据集划分、自定义模型、模型训练、激活函数使用、优化器使用，示例基于CIFAR10数据集，pytorch框架实现。

## 1、关键流程代码说明

### 1.1、数据加载

```python
def dataloader(self):
	'''
	加载CIFAR10数据
	:return:
	'''
	train_data = CIFAR10(root='./data/train/', 
                         train=True, 
                         download=True, 
                         transform=TransformUtils.get_train_transforms())
        self.show_img(train_data)
        test_data = CIFAR10(root='./data/test/', 
                            train=False, 
                            download=True, 
                            transform=TransformUtils.get_test_transforms())
        train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_data, batch_size=100, shuffle=False, num_workers=2)
        return train_loader, test_loader
```

- 该dataloader函数是自己写的，基于CIFAR10数据集和pytorch框架的DataLoader类实现。
- 函数中先通过CIFAR10下载训练集和测试集数据，分别存于/data/train/和/data/test/目录下面，同时对图片进行了transform转化，后面会讲自定义的transform
- 下载后的数据通过DataLoader类进行封装，DataLoader可以设置批量大小batch_size，是否混淆打乱shuffle以及处理的进程数num_workers

### 1.2、自定义模型

```python
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
le_model = LeNet5().to(device=device)
```

> 该自定义模型LeNet5继承了pytorch的nn.Module类，实现了2个卷积层和三个全连接层的网络。
>
> 个人感觉前向传播函数的本质就是通过y=f(x)这样的函数关系式子把初始化的值串起来。
>
> 串联过程中卷积层1和卷积层2分别使用了relu激活函数并通过最大池化函数进行汇聚。
>
> 然后通过x.view对形状进行重塑。
>
> 最后，三个中的前两个全连接层使用了relu激活函数。

### 1.3、训练

```python
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
```

> 训练的前置准备条件：
>
> - 1、实例化交叉熵损失函数
> - 2、实例化优化器，本例中使用的优化器是SGD
> - 3、实例化自定义模型
> - 4、通过自定义dataloader获取训练集和测试集
> - 5、设置迭代次数epoch

**核心训练过程：**

- 1、从train_loader中取出数据和标签
- 2、把数据放到GPU上
- 3、使用优化器对梯度进行清零
- 4、把数据喂给模型
- 5、计算损失
- 6、对损失进行反向传播计算
- 7、进行梯度下降计算
- 8、获取本轮的损失值

### 1.4、保存模型

```python
    def save_model(self):
        '''
        保存模型
        :return:
        '''
        model = self.get_model()
        torch.save(model.state_dict(), './data/models/model.pth')
```

本函数实现过程：

- 1、获取模型
- 2、保存模型

### 1.5、测试

```python
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
```

测试验证模型的准确性，先通过自定义的dataloader获取测试数据，然后加载获取模型，接着对测试集的数据对进行迭代，在迭代中进行预测和准确率的计算。另外在进行迭代过程中跟训练时一样，先把数据放到GPU中，把测试数据喂给模型获取预测值。
