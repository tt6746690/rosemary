import itertools
import os
import random
import numpy as np
import torch

__all__ = [
    'torch_unnormalize',
    'torch_tensor_to_ndarray',
    'torch_cat_dicts',
    'torch_set_random_seed',
    'torch_configure_cuda',
    'torch_input_grad',
    'torch_get_dimensions',
]


def torch_unnormalize(tensor, mean=0.628, std=0.255):
    """Reverse `Normalize(mean, std)` for plotting purposes. """
    dtype, device = tensor.dtype, tensor.device
    mean = torch.as_tensor(mean, dtype=dtype, device=device)
    std = torch.as_tensor(std, dtype=dtype, device=device)
    if mean.ndim == 1: mean = mean.view(-1, 1, 1)
    if std.ndim == 1: std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


def torch_get_dimensions(img):
    """Introduced in torchvision=1.13.0."""
    from PIL import Image
    if isinstance(img, Image.Image):
        if hasattr(img, "getbands"):
            channels = len(img.getbands())
        else:
            channels = img.channels
        width, height = img.size
        return [channels, height, width]
    if isinstance(img, (torch.Tensor, np.ndarray)):
        channels = 1 if img.ndim == 2 else img.shape[-3]
        height, width = img.shape[-2:]
        return [channels, height, width]


def torch_tensor_to_ndarray(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().to('cpu').numpy()
    return x


def torch_cat_dicts(L):
    """Concatenates a list of dictionary of torch tensors
            L = [{k: v1}, {k: v2}, ...]       to {k: torch.as_tensor([v1; v2])}
            L = [{k: [v1,v2]}, {k: [v3,v4]}, ...] to {k: [v1;v2;v3;v4]} 
            
        ```
        a = torch.as_tensor([1,2])
        b = torch.as_tensor([[1,2,3,4],[5,6,7,8]])
        c = torch.tensor(1)
        e = {'a': a, 'b': b, 'c': c, 'a_list': list(a), 'b_list': list(b)}
        L = [e,e]
        torch_cat_dicts(L)
        # {'a': tensor([1, 2, 1, 2]),
        #  'b': tensor([[1, 2, 3, 4],
        #          [5, 6, 7, 8],
        #          [1, 2, 3, 4],
        #          [5, 6, 7, 8]]),
        #  'c': tensor([1, 1]),
        #  'a_list': [tensor(1), tensor(2), tensor(1), tensor(2)],
        #  'b_list': [tensor([1, 2, 3, 4]),
        #   tensor([5, 6, 7, 8]),
        #   tensor([1, 2, 3, 4]),
        #   tensor([5, 6, 7, 8])]}
        ```
    """
    d = {}
    if L:
        K = L[0].keys()
        for k in K:
            batch_is_a_list = isinstance(L[0][k], list)
            if batch_is_a_list:
                elem_ndim = list(L[0][k])[0].ndim
                cat_fn = list
            else:
                elem_ndim = L[0][k].ndim
                cat_fn = torch.hstack if elem_ndim == 0 else torch.cat
            tensors = [x[k] for x in L]
            if batch_is_a_list:
                tensors = list(itertools.chain.from_iterable(tensors))
            d[k] = cat_fn(tensors)
    return d


def torch_set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def torch_configure_cuda(gpu_id):
    # some bug in `DataLoader` spawn ~16 unnecessary threads
    # that cogs up every single CPU and slows downs data loading
    # Set number of threads to 1 to side-step this problem.
    torch.set_num_threads(1)

    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    if not torch.cuda.is_available():
        raise AssertionError(f'GPU not available!')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def torch_input_grad(f, model, x):
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
