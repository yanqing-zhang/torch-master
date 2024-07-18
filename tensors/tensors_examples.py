'''
@Project ：torch-master 
@File    ：tensors_examples.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/7/17 17:59 
'''
import torch
import numpy as np

class TensorsExamples:

    def simple_cpu_example(self):
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        y = torch.tensor([[7, 8, 9], [10, 11, 12]])
        z = x + y
        print(f"z:{z}")
        print(f"size of z:{z.size()}")

    def simple_gpu_example(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
        y = torch.tensor([[7, 8, 9], [10, 11, 12]], device=device)
        z = x + y
        print(f"z:{z}")
        print(f"size of z:{z.size()}")
        print(f"device:{device}")

    def move_tensor_cpu_gpu_example(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
        y = torch.tensor([[7, 8, 9], [10, 11, 12]], device=device)
        z = x + y
        z.to("cpu")
        print(f"z:{z}")

    def create_tensors(self):
        '''
        创建张量
        :return:
        '''
        a = torch.tensor([1, 2, 3])
        b = torch.tensor((1, 2, 3))
        c = torch.tensor(np.array([1, 2, 3]))

        d = torch.empty(100, 200) # 创建100×200的空张量，每一个元素都为0
        e = torch.zeros(100, 200) # 创建100×200的空张量，每一个元素都为0
        f = torch.ones(100, 200) # 创建100×200的张量，每一个元素都为1
        print(f"a:{a}\n")
        print(f"b:{b}\n")
        print(f"c:{c}\n")
        print(f"size of d:{d.size()} \n d:{d}\n")
        print(f"size of e:{e.size()} \n e:{e}\n")
        print(f"size of f:{f.size()} \n f:{f}\n")

    def create_tensors_by_rand(self):
        '''
        创建随机张量
        :return:
        '''
        a = torch.rand(100, 200)
        b = torch.randn(100, 200)
        c = torch.randint(5, 10, (100, 200))
        d = torch.empty((100, 200), dtype=torch.float64, device="cuda")
        x = torch.empty_like(d)
        print(f"size of a:{a.size()} \n a:{a}\n")
        print(f"size of b:{b.size()} \n b:{b}\n")
        print(f"size of c:{c.size()} \n c:{a}\n")
        print(f"size of d:{d.size()} \n d:{d}\n")
        print(f"size of x:{x.size()} \n x:{x}\n")

    def tensors_operate(self):
        '''
        操作张量
        :return:
        '''
        x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        print(f"x:{x}")
        print("\n")
        print(x[1, 1].item()) #第二行第二列的值: 4
        print("\n")
        print(x[:2, 1]) # 切片：tensor([2, 4])
        print(x[x<5]) #把元素小于5的找到列出来
        print(x.t()) # 行列转置
        print(x.view((2, 4)))
        print(x.view((-1, 1))) # -1代表任意，x.view((-1, 1))表示一列任意行
        y = torch.stack((x, x)) # 堆叠
        print(f"y:{y}")
        a, b = x.unbind(dim=1) #按列进行拆分，因为x只有两列，所以拆分后得a,b
        print(a, b)

    def tensors_autograd(self):
        '''
        自动求导
        :return:
        '''
        x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float, requires_grad=True)
        print(f"x:{x}")
        f = x.pow(2).sum()
        print(f"f:{f}")
        f.backward()
        print(f"grad:{x.grad}")
if __name__ == '__main__':
    t = TensorsExamples()
    if False:
        t.simple_cpu_example()
        t.simple_gpu_example()
        t.move_tensor_cpu_gpu_example()
        t.create_tensors()
        t.create_tensors_by_rand()
        t.tensors_operate()
    else:
        t.tensors_autograd()