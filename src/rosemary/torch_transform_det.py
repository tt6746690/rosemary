## Modified from https://github.com/facebookresearch/detr/blob/a54b77800e/datasets/transforms.py
#  - remove `target.copy()`, since the underlying torch tensor is not cloned anyway.
#  - create transformations that applies to `target` only.
#
# Note names not exported to the global scope `rosemary` by default.
# Use `import rosemary.torch_transform_det as T_det` to access function.
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import numbers
import random

import torch

# needed due to empty tensor bug in pytorch and torchvision 0.5
from packaging import version
import torchvision
if version.parse(torchvision.__version__) < version.parse("0.7.0"):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size
    
import torchvision.transforms as T
import torchvision.transforms.functional as F

from .torch import torch_get_dimensions

## Deal with legacy code where the `InterpolationMode`
#  does not exist yet.
if not hasattr(F, 'InterpolationMode'):
    from enum import Enum
    class InterpolationMode(Enum):
        NEREAST  = 0
        BILINEAR = 2
        BICUBIC  = 3
    setattr(F, 'InterpolationMode', InterpolationMode)


__all__ = [
    "crop_target",
    "center_crop_target",
    "center_crop",
    "hflip",
    "hflip_target",
    "resize",
    "pad",
    "RandomCrop",
    "RandomSizeCrop",
    "CenterCrop",
    "RandomHorizontalFlip",
    "RandomResize" 
    "RandomPad",
    "RandomSelect",
    "ToTensor",
    "RandomErasing",
    "Normalize",
    "Compose",
    "Lambda",
    "LambdaImage",
    "LambdaTarge",
    "ExpandChannels",
]


# from torchvision.transforms.util.box_ops import box_xyxy_to_cxcywh
# from torchvision.transforms.util.misc import interpolate


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    ## I'm guessing minor version be less than <0.7
    #  commented code doesn't work for `0.13.0`
    # if float(torchvision.__version__[:3]) < 0.7:
    if version.parse(torchvision.__version__) < version.parse("0.7.0"):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )
        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def crop(image, target, region):
    cropped_image = F.crop(image, *region)
    target = crop_target(target, region)
    return cropped_image, target


def crop_target(target, region):
    # Note `.copy()` doesn't clone the underlying tensor, just the pointer.
    # target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]
        if "text" in target:
            target["text"] = [x for (x, b) in zip(target["text"], keep) if b]
    return target


def center_crop_target(target, output_size, image_shape):
    region = center_crop_region(image_shape, output_size)
    target = crop_target(target, region)
    return target


def center_crop_region(image_shape, output_size):
    """Modified from https://pytorch.org/vision/main/_modules/torchvision/transforms/functional.html#center_crop 
            `output_size` in (h, w)
            `image_shape` in (h, w)
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])
    image_height, image_width = image_shape
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    region = crop_top, crop_left, crop_height, crop_width
    return region


def center_crop(image, target, output_size):
    region = center_crop_region(torch_get_dimensions(image)[1:], output_size)
    return crop(image, target, region)


def hflip(image, target):
    flipped_image = F.hflip(image)
    w = torch_get_dimensions(image)[2]
    target = hflip_target(target, w)
    return flipped_image, target


def hflip_target(target, image_width):
    # target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes[:, [0, 2]] = image_width - boxes[:, [2, 0]]
        target["boxes"] = boxes
    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)
    return target


def resize(image, target, size, interpolation=F.InterpolationMode.BILINEAR, max_size=None):
    # size can be min_size (scalar) or (h, w) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(get_image_size(image), size, max_size) # (h, w)
    rescaled_image = F.resize(image, size, interpolation=interpolation, max_size=max_size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig 
        in zip(get_image_size(rescaled_image), get_image_size(image)))
    ratio_width, ratio_height = ratios

    # target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    # target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(torch_get_dimensions(padded_image)[1:])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        region = T.RandomCrop.get_params(image, self.size)
        return crop(image, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        image_width, image_height = get_image_size(image)
        w = random.randint(self.min_size, min(image_width, self.max_size))
        h = random.randint(self.min_size, min(image_height, self.max_size))
        region = T.RandomCrop.get_params(image, [h, w])
        return crop(image, target, region)


class CenterCrop(T.CenterCrop):
    def __init__(self, size):
        super().__init__(size)

    def forward(self, image, target):
        return center_crop(image, target, self.size)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            return hflip(image, target)
        return image, target


class Resize(T.Resize):
    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__(size, interpolation=interpolation, max_size=max_size, antialias=antialias)
        
    def forward(self, image, target):
        return resize(image, target, self.size, self.interpolation, self.max_size)


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, image, target=None):
        size = random.choice(self.sizes)
        return resize(image, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, image, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(image, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            return self.transforms1(image, target)
        return self.transforms2(image, target)


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, image, target):
        return self.eraser(image), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        # target = target.copy()
        ## wpq: used for training detectors. shouldn't be in this class.
        # h, w = image.shape[-2:]
        # if "boxes" in target:
        #     boxes = target["boxes"]
        #     boxes = box_xyxy_to_cxcywh(boxes)
        #     boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
        #     target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Lambda(object):
    def __init__(self, lambd):
        self.lambd = lambd
    
    def __call__(self, image, target):
        return self.lambd(image, target)


class LambdaImage(T.Lambda):
    def __init__(self, lambd):
        super().__init__(lambd)
    
    def __call__(self, image, target):
        image = super().__call__(image)
        return image, target
    

class LambdaTarget(T.Lambda):
    def __init__(self, lambd):
        super().__init__(lambd)
    
    def __call__(self, image, target):
        target = super().__call__(target)
        return image, target


class ExpandChannels:
    """ #channels 1 -> 3 via copy. """

    def __call__(self, image, target):
        dims = torch_get_dimensions(image)
        if dims[0] != 1:
            raise ValueError(
               f"Number of channels should be 1 but got {dims[0]}")
        image = image.reshape(dims)
        image = torch.repeat_interleave(image, 3, dim=0)
        return image, target


def get_image_size(image):
    return torch_get_dimensions(image)[1:][::-1]