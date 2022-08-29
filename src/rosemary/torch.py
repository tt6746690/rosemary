from typing import List, Dict
import dataclasses

import itertools
import os
import random
import numpy as np
import torch

__all__ = [
    'torch_unnormalize',
    'torch_tensor_to_ndarray',
    'torch_combine_batches',
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

def torch_non_batch_dims_match(L):
    """Return True if list of Tensor has same non-batch dimensions."""
    non_batch_dims = [tuple(x.shape[1:]) for x in L]
    shapes_match = all(x == non_batch_dims[0] for x in non_batch_dims)
    return shapes_match


def torch_combine_batches(L, apply_to_tensor_fn=None):
    """Combine `[b, ...]` where `b` is a batch.
        This function differs from `default_collate` in that it **avoids**
            - adding batch dimension.
            - avoids converting to `torch.Tensor` when possible
        Also handles the case where `b` is a `transformers.ModelOutput`
            
        For example:
            L = [{k: v1}, {k: v2}, ...]           -> {k: torch.as_tensor([v1; v2])}
            L = [{k: [v1,v2]}, {k: [v3,v4]}, ...] -> {k: [v1;v2;v3;v4]} 

        ```
        a = torch.as_tensor([1,2])
        b = torch.as_tensor([[1,2,3,4],[5,6,7,8]])
        c = torch.tensor(1)
        d = torch.as_tensor([[1,2]])
        e = {'a': a, 'b': b, 'c': c, 'd': d ,
            'a_list': list(a), 'b_list': list(b), 'dd_tuple': tuple([d,d+10]),}
        L = [e,e]
        torch_combine_batches(L)
        # {'a': tensor([1, 2, 1, 2]),
        # 'b': tensor([[1, 2, 3, 4],
        #         [5, 6, 7, 8],
        #         [1, 2, 3, 4],
        #         [5, 6, 7, 8]]),
        # 'c': tensor([1, 1]),
        # 'd': tensor([[1, 2],
        #         [1, 2]]),
        # 'a_list': [tensor(1), tensor(2), tensor(1), tensor(2)],
        # 'b_list': [tensor([1, 2, 3, 4]),
        # tensor([5, 6, 7, 8]),
        # tensor([1, 2, 3, 4]),
        # tensor([5, 6, 7, 8])],
        # 'aa_tuple': (tensor([[1, 2],
        #         [1, 2]]),
        # tensor([[11, 12],
        #         [11, 12]]))}

        ```
        # test `ModelOutput`
        from rosemary import torch_combine_batches
        L = [{'a': outputs}, {'a': outputs}]
        o = torch_combine_batches(L)
        o = o['a']
        print('o.image_embeds', [(k,tuple(v.shape)) for k, v in o.image_embeds])
        print('o.text_embeds', tuple(o.text_embeds.shape))
        fpnoutput = o.vision_model_output
        print('o.fpn_output', [(k,v.shape) for k,v in fpnoutput.fpn_output])
        for k, v in fpnoutput.vision_model_output.items():
            shapes = v.shape if isinstance(v, torch.Tensor) else [tuple(x.shape) for x in v]
            print(f'o.vision_model.{k}', shapes)
        for k, v in o.text_model_output.items():
            shapes = v.shape if isinstance(v, torch.Tensor) else [tuple(x.shape) for x in v]
            print(f'o.text_model.{k}', shapes)
        ```
    """
    apply_to_tensor_list_fn = lambda L: \
        [apply_to_tensor_fn(x) for x in L] if (apply_to_tensor_fn is not None) and \
            all(isinstance(Li, torch.Tensor) for Li in L) else L

    is_list_of_two_tuple = lambda l: \
        isinstance(l, list) and all((isinstance(li, tuple) and len(li)==2) for li in l)
    is_tuple_of_tensor = lambda l: \
        isinstance(l, tuple) and all(isinstance(li, torch.Tensor) for li in l)

    b = L[0]
    if isinstance(b, torch.Tensor):
        if apply_to_tensor_fn is not None:
            L = [apply_to_tensor_fn(x) for x in L]
        if b.ndim == 0:
            return torch.hstack(L)
        else:
            # handles List[Tensor] with varying shapes, e.g., seq_len different in each batch.
            cat_fn = torch.cat if torch_non_batch_dims_match(L) else list
            return cat_fn(L)
    elif isinstance(b, list):
        L = list(itertools.chain.from_iterable(L))
        L = apply_to_tensor_list_fn(L)
        return L
    elif all(is_tuple_of_tensor(l) for l in L):
        # tuple indicates a single sample/instance, unlike list.
        # Didn't account case when tensor.ndim == 0!
        def _cat_fn(k):
            q = [li[k] for li in L]
            q = apply_to_tensor_list_fn(q)
            # take into account variable shapes when doing torch.cat
            return torch.cat(q) if torch_non_batch_dims_match(q) else q
        L = tuple(_cat_fn(k) for k in range(len(L[0])))
        return L
    # try to reduce extra dependency on `transformers` package.
    # elif isinstance(b, ModelOutput):
    elif isinstance(b, dict) and dataclasses.is_dataclass(b):
        cls = type(b)
        o = {}
        for k in b.keys():
            l = [b[k] for b in L]
            # Note we can not move these cases to base case of this function
            # since the behavior is different when inside/outside `ModelOutput`
            
            if all(is_list_of_two_tuple(li) for li in l):
                # special case where l := [[(k, v1), ...], ...]
                # consider `l` as a dictionary -> [(k, [v1;v2;...])]
                l: List[Dict] = [dict(x) for x in l] # combine batches 
                l = torch_combine_batches(l, apply_to_tensor_fn)
                l = list(zip(l.keys(), l.values()))  # return to 2-tuple repr.
                o[k] = l
            elif all(is_tuple_of_tensor(li) for li in l):
                # special case when l := [(v1,v2,...),(va,vb,...),...] where values are tensors.
                # do not combine `l` as [v1,v2,...,va,vb,...].
                # consider `l` as a dictionary -> [cat(v1;va,...), cat(v2;vb,...)]
                def _cat_fn(k):
                    q = [li[k] for li in l]
                    q = apply_to_tensor_list_fn(q)
                    # take into account variable shapes when doing torch.cat
                    return torch.cat(q) if torch_non_batch_dims_match(q) else q
                l = tuple(_cat_fn(k) for k in range(len(l[0])))
                o[k] = l
            else:
                l = torch_combine_batches(l, apply_to_tensor_fn)
                o[k] = l
        return cls(**o)
    elif isinstance(b, dict):
        d = {}
        for k in b.keys():
            l = [b[k] for b in L]
            d[k] = torch_combine_batches(l, apply_to_tensor_fn)
        return d
    else:
        raise ValueError(f'`combine_batches` does not support {type(b)}')


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
