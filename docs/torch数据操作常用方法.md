# torch数据操作常用方法

## 0、说明

本笔记会把常用api练习一遍，覆盖深度学习全流程。

## 1、前置条件

在使用pytorch进行深度学习开发时，需要提前安装好开发环境：

- anaconda安装好 （略，别的文档有记录）
- cudnn安装好（略，别的文档有记录）
- pycharm安装好（略，别的文档有记录）

## 2、torch相关安装
```python
# 安装torch、torchvion、torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 安装常用组件matplotlib/pandas/seaborn
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 3、张量创建与维度识别

### 3.1、torch.randn重要

```python
import torch

a = torch.randn(2,3,4)
print(f"a.shape:{a.shape}")
```

torch.randn函数用于生成一个符合标准正态分布（均值为0，方差为1）的随机张量。该函数中的入参2，3，4表示的是张量的形状。

- **第一个维度（2）**：表示张量中有2个“块”或“通道”。
- **第二个维度（3）**：表示每个“块”中有3个“行”。
- **第三个维度（4）**：表示每个“行”中有4个“列”或元素。

整个torch.randn(2, 3, 4)函数将产生 `2 * 3 * 4 = 24`个随机生成的元素。

### 3.2、x.view重要

```python
x = torch.tensor([[1,2],[3,4],[5,6],[7,8]])
m = x.view((2, 4)) # 将上面的x(4行2列)，转成2行4列，入参表示的是形状值行与列
n = x.view((-1,1)) # 固定列数为1，行数自动计算,此法在模型的框架修改时，特别是在最后一层全连接层的修改时会用到此函数
```

### 3.3、x.reshape重要

```python
x = torch.tensor([[1,2],[3,4],[5,6],[7,8]])
p = x.reshape((-1, 1))
k = x.reshape((8, 1))
# p与k的效果一样，但用法不一样，p是固定列数1，行自动计算，而k是行列都固定，此函数如果行数定义超出范围或少于最大值时会报错
```

### 3.4、torch.stack重要

```python
x = torch.tensor([[1,2],[3,4],[5,6],[7,8]])
o = torch.stack((x, x)) # 把x与x进行叠加输出，此函数在模型构建和修改时也非常有用。
```

### 3.5、x.unbind重要

```python
x = torch.tensor([[1,2],[3,4],[5,6],[7,8]])
q,r = x.unbind(dim=1) # 将张量x拆分成2个张量，在dim=1即按列的维度进行拆分[1,3,5,7],[2,4,6,8]
```

### 3.6、获取指定位置值

```python
x = torch.tensor([[1,2],[3,4],[5,6],[7,8]])
x1 = x[1, 1] # out:取第二行第二列的元素张量tensor(4) 
x11 = x[1, 1].item() # out:取第二行第二列的元素张量的值: 4
```

### 3.7、切片

```python
x = torch.tensor([[1,2],[3,4],[5,6],[7,8]])
a = x[:2, 1] # out:输出前两行的第2列:[2,4]
b = x[x<5] # out:输出所有元素小于5的值[1,2,3,4] ,这个条件不管行与列，面向所有元素
```

### 3.8、各种创建张量

```python
# Created from pre-existing arrays
w = torch.tensor([1,2,3]) # 1维，通过列表[1,2,3]创建张量，入参即为张量元素内容
w = torch.tensor((1,2,3)) # 1维，通过元组(1,2,3)创建张量，入参即为张量元素内容
w = torch.tensor(numpy.array([1,2,3])) # 1维，通过数组numpy.array([1,2,3])创建张量，入参即为张量元素内容

# Initialized by size
w = torch.empty(100,200) # 2维(因为两个表示形状的入参), 100 * 200 = 20000个元素,各元素默认值为0
w = torch.zeros(100,200) # 2维(因为两个表示形状的入参), 100 * 200 = 20000个元素,各元素默认值为0
w = torch.ones(100,200)  # 2维(因为两个表示形状的入参), 100 * 200 = 20000个元素,各元素默认值为1
i = torch.empty_like(e)  # 复制torch.empty结果
j = torch.zeros_like(f)  # 复制torch.zeros结果
k = torch.ones_like(g)  # 复制torch.ones结果
```

### 3.9、设置数据类型

```python
a = torch.tensor([1,2,3], dtype=torch.float32) # torch.int32,torch.float32,torch.float64...
print(f"a.dtype:{a.dtype}") # out:torch.float32
```

### 3.10、求导

```python
x = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float, requires_grad=True) # 先设置变量可导
s = x.pow(2).sum()  # 将x中的每个元素求平方，然后把平方后的值累加求和，此时构建了一个式子 s = x^2
print(s) # tensor(91., grad_fn=<SumBackward0>)
s.backward() # 反向传播
r = x.grad # ds/dx = 2x 求导
```

求导操作首先要设置变量为可导，然后再进行反向传播和求导计算。

### 3.11、统计张量元素个数

```python
x = torch.rand(100, 200)
a = x.numel()
print(f"a:{a}")
```

> x.numel()计算了张量x中元素的总数,numel是"number of elements"的缩写，这个方法返回张量中元素的总数

### 3.12、N维数组样例

<img src="./images\image-20240827212437968.png" alt="image-20240827212437968" style="zoom:67%;" /> 	

<img src="./images\image-20240827212501429.png" alt="image-20240827212501429" style="zoom:67%;" />

### 3.13、创建数组

<img src="./images\image-20240827212630686.png" alt="image-20240827212630686" style="zoom:67%;" />


### 3.14、元素访问

<img src="./images\image-20240827212703516.png" alt="image-20240827212703516" style="zoom:67%;" />
