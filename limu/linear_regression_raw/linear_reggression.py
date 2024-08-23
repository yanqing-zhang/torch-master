import torch
import random

class LinearReggressionRaw:
    """
    从零实现线性回归
    """
    def synthetic_data(self, w, b, num_examples):
        """
        异常:RuntimeError: Trying to backward through the graph a second time
         (or directly access saved tensors after they have already been freed).
         Saved intermediate values of the graph are freed when you call .
         backward() or autograd.grad(). Specify retain_graph=True
         if you need to backward through the graph a second time or
         if you need to access saved tensors after calling backward.
         原因是 l.sum().backward()这里改成 l.sum().backward(retain_graph=True)
        生成数据时不能设置可导requires_grad=True
        训练时w,b需要设置，这两点不遵循都会报错上面的错误
        """
        X = torch.normal(0, 1, (num_examples, len(w)))  # 生成 X，他是一个均值为0，方差为1的随机数，他的大小: 行为num_examples，列为w的长度表示多少个feature
        y = torch.matmul(X, w) + b
        y += torch.normal(0, 0.01, y.shape)  # 加入一些噪音，均值为0 ，方差为0.01，形状和y是一样
        return X, y.reshape((-1, 1))  # 把X和y作为一个列向量返回

    def data_iter(self, batch_size, features, labels):
        """
        当函数使用yield时，它就变成了一个生成器函数。每次调用生成器的__next__()方法或
        使用for循环迭代时，函数会执行到yield语句并返回一个值，然后函数的状态（包括局部变量）会被冻结保存起来，
        直到下一次调用。
        """
        len_features = len(features)
        indexs = list(range(len_features))
        random.shuffle(indexs)

        for i in range(0, len_features, batch_size):
            batch_indexs = torch.tensor(indexs[i: min(i + batch_size, len_features)])
            yield features[batch_indexs], labels[batch_indexs]

    def print_data_iter(self, w, b, num_examples):
        """
        利用自定义的数据迭代器打印自定义的数据
        """
        batch_size = 10
        feature, labels = self.synthetic_data(w, b, num_examples)
        for X, y in self.data_iter(batch_size, features=feature, labels=labels):
            print(f"X:{X}, y:{y}")
            break

    def get_w_b(self):
        """
        初始化模型参数
        """
        w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)  # w:size为2行1列,随机初始化成均值为0，方差为0.01的正态分布，requires=true是指需要计算梯度
        b = torch.zeros(1, requires_grad=True)  # 对于偏差来说直接为0，1表示为一个标量，因为也需要进行更新所以为True
        return w, b

    def line_reggression_model(self, X, w, b):
        """
        定义基本的线性回归模型
        """
        return torch.matmul(X, w) + b    #矩阵乘以向量再加上偏差

    def squared_loss(self, y_hat, y):
        """
        损失函数：均方误差
        """
        return (y_hat - y.reshape(y_hat.shape))**2 / 2      #按元素做减法，按元素做平方，再除以2  （这里没有做均方）
    def sgd(self, params, lr, batch_size):
        """
        自定义优化器
        torch.no_grad()：不要计算和存储梯度（gradients）
        params：该参数是损失
        param.grad.zero_()：执行完一轮梯度后，要把梯度清零，否则会累加
        """
        with torch.no_grad():  # 这里更新的时候不需要参与梯度计算所以是no_grad
            for param in params:  # 对于参数中的每一个参数，可能是w可能是b
                param -= lr * param.grad / batch_size  # 参数减去learning rate乘以他的梯度（梯度会存在.grad中）。上面的损失函数中没有求均值，所以这里除以了batch_size求均值，因为乘法对于梯度是一个线性的关系，所以除以在上面损失函数那里定义和这里是一样的效果
                param.grad.zero_()  # 把梯度设置为0，因为pytorch不会自动的设置梯度为0，需要手动，下次计算梯度的时候就不会与这次相关了

    def fit(self, batch_size, features, labels, w, b):
        lr = 0.03
        epochs = 10
        net = self.line_reggression_model
        loss = self.squared_loss

        for epoch in range(epochs):  # 训练的过程基本是两层for循环（loop）,第一次for循环是对数据扫一遍
            for X, y in self.data_iter(batch_size, features, labels):  # 对于每一次拿出一个批量大小的X和y
                l = loss(net(X, w, b), y)  # 把X,w,b放进network中进行预测，把预测的y和真实的y来做损失，则损失就是一个长为批量大小的一个向量，是X和y的小批量损失
                # l(loss)的形状是（'batch_size',1）,而不是一个标量
                l.sum().backward()  # 对loss求和然后算梯度。计算关于['w','b']的梯度
                self.sgd([w, b], lr, batch_size)  # 算完梯度之后就可以访问梯度了，使用sgd对w和b进行更新。使用参数的梯度对参数进行更新
            # 对数据扫完一遍之后来评价一下进度，这块是不需要计算梯度的，所以放在no_grad里面
            with torch.no_grad():
                train_l = loss(net(features, w, b), labels)  # 把整个features，整个数据传进去计算他的预测和真实的labels做一下损失，然后print
                print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


if __name__ == '__main__':
    lrr = LinearReggressionRaw()
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = lrr.synthetic_data(true_w, true_b, 1000)
    w, b = lrr.get_w_b()
    num_examples = 10
    lrr.print_data_iter(w, b, num_examples)
    batch_size = 10

    lrr.fit(batch_size, features, labels, w, b)