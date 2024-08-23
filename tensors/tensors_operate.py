'''
@Project ：torch-master 
@File    ：tensors_operate.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/8/19 13:42 
'''
import torch

class TensorsOperate:

    def get_tensor(self):
        x = torch.tensor([1.0, 2, 4, 8])
        y = torch.tensor([2, 2, 2, 2])
        return  x, y

    def base_operation(self):
        x, y = self.get_tensor()
        a = x + y
        b = x - y
        c = x * y
        d = x / y
        e = x ** y # 次方
        f = torch.exp(x)
        print(f"a:{a}, b:{b}, c:{c}, d:{d}, e:{e}, f:{f}")

if __name__ == '__main__':
    t = TensorsOperate()
    if False:
        print("---------")
    else:
        t.base_operation()

