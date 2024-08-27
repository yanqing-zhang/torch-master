import torch

x = torch.arange(12)
print(f"x:{x}")
print(f"x.shape:{x.shape}")
print(f"x.numel:{x.numel()}")

X = x.reshape(3, 4)
print(f"X:{X}")

a = torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
print(f"a:{a}")