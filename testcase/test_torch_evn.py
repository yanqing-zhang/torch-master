import torch

def show_torch_info():
    print(f"torch.version:{torch.__version__}")
    print(f"cuda is available:{torch.cuda.is_available()}")

if __name__ == '__main__':
    show_torch_info()