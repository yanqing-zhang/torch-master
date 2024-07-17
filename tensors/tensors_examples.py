'''
@Project ：torch-master 
@File    ：tensors_examples.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/7/17 17:59 
'''
import torch


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
if __name__ == '__main__':
    t = TensorsExamples()
    if False:
        t.simple_cpu_example()
    else:
        t.simple_gpu_example()