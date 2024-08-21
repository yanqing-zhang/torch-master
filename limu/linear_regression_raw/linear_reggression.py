import torch
import random
from d2l import torch as d2l

class LinearReggressionRaw:
    def synthetic_data(w, b, num_examples):
        X = torch.normal(0, 1, (num_examples, len(w)))
        y = torch.matmul(X, w) + b
        y += torch.normal(0, 0.01, y.shape)
        return X, y.reshape((-1, 1))