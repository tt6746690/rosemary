import os
import random
import numpy as np
import torch

__all__ = [
    'torch_tensor_to_ndarray',
    'torch_cat_dicts',
    'torch_set_random_seed',
    'torch_configure_cuda',
    'torch_input_grad',
    'torch_get_dimensions',
]


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
            L = [{k: v1}, {k: v2}, ...] to {k: [v1; v2]} """
    d = {}
    if L:
        K = L[0].keys()
        cat_fns = [torch.hstack if L[0][k].ndim == 0 else torch.cat
                   for k in K]
        for cat_fn, k in zip(cat_fns, K):
            d[k] = cat_fn([x[k] for x in L])
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
