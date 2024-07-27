'''
@Project ：torch-master 
@File    ：tensor_dim_example.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/7/24 9:27 
'''
import torch
import numpy as np

a = torch.randn(2, 3, 4) # 3维, 元素个数:2 * 3 * 4 = 24个，各元素随机数符合正态分布
print(f"a:{a}")
print(f"a.shape:{a.shape}")
print("--------------------------------")
b = torch.randn(2, 3)
print(f"b:{b}")
print(f"b.shape:{b.shape}")
print("--------------------------------")
c = torch.tensor([1,2,3]) # 1维，通过列表创建张量
print(f"c:{c}")
print(f"c.shape:{c.shape}")
print("--------------------------------")
d = torch.tensor((1,2,3)) # 1维，通过元组创建张量
print(f"d:{d}")
print(f"d.shape:{d.shape}")
print("--------------------------------")
e = torch.empty(100,200) # 2维, 100 * 200 = 20000个元素,各元素默认值为0
print(f"e:{e}")
print(f"e.shape:{e.shape}")
print("--------------------------------")
f = torch.zeros(100,200) # 2维, 100 * 200 = 20000个元素,各元素默认值为0
print(f"f:{f}")
print(f"f.shape:{f.shape}")
print("--------------------------------")
g = torch.ones(100,200) # 2维, 100 * 200 = 20000个元素,各元素默认值为1
print(f"g:{g}")
print(f"g.shape:{g.shape}")
print("--------------------------------")
h = torch.tensor(np.array([1,2,3]))
print(f"h:{h}")
print(f"h.shape:{h.shape}")
print("---------------------------------")
x = torch.tensor([[1,2],[3,4],[5,6],[7,8]])
print(f"x:{x}")
print(f"x.shape:{x.shape}")
print("---------------------------------")
i = torch.empty_like(e)
print(f"i:{i}")
print(f"i.shape:{i.shape}")
print("---------------------------------")
j = torch.zeros_like(f)
print(f"j:{j}")
print(f"j.shape:{j.shape}")
print("---------------------------------")
k = torch.ones_like(g)
print(f"k:{k}")
print(f"k.shape:{k.shape}")
print("---------------------------------")
m = torch.tensor([[1,2],[3,4],[5,6],[7,8]])
l = m.view((2,4))
print(f"l:{l}")
print(f"l.shape:{l.shape}")
print("---------------------------------")
n = m.view((-1,1)) # 固定列数为1，行数自动计算
print(f"n:{n}")
print(f"n.shape:{n.shape}")
print("---------------------------------")
o = torch.stack((x, x))
print(f"o:{o}")
print(f"o.shape:{o.shape}")
print("---------------------------------")
x = torch.tensor([[1,2],[3,4],[5,6],[7,8]])
p = x.reshape((-1, 1))
print(f"p:{p}")
print(f"p.shape:{p.shape}")
print("---------------------------------")
q,r = x.unbind(dim=1)
print(f"q:{q}")
print(f"q.shape:{q.shape}")
print(f"r:{r}")
print(f"r.shape:{r.shape}")
print("---------------------------------")
x = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float, requires_grad=True)
s = x.pow(2).sum()
print(s) # tensor(91., grad_fn=<SumBackward0>)
s.backward()
r = x.grad # ds/dx = 2x
print(r)
# tensor([[ 2.,  4.,  6.],
#         [ 8., 10., 12.]])