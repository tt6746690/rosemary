import os

import numpy as np
import torch


def torch_set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def torch_configure_cuda(gpu_id):
    # some bug in `DataLoader` spawn ~18 unnecessary threads
    # that cogs up every single CPU and slows downs data loading
    # Set number of threads to 1 to side-step this problem
    torch.set_num_threads(1)

    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    if not torch.cuda.is_available():
        raise AssertionError(f'GPU not available!')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def input_grad(f, model, x):
    # Computes Gradient of `f(model, x)`` w.r.t. `x`
    # 
    with torch.enable_grad():
        model.zero_grad()
        x = x.detach()          # Creates a tensor with shared storage
        x.requires_grad = True  # Add input to computation graph

        L = f(model, x)
        L.backward()            # Computes gradient

        g = x.grad.detach()
        model.zero_grad()

    return g