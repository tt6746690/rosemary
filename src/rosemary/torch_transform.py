from PIL import Image

import numpy as np
import torch
from torchvision.transforms import Resize
from torchvision.transforms.functional import resize, InterpolationMode

__all__ = [
    'GrayscaleJitter',
    'resize_with_clip',
    'ResizeWithClip',
]


def resize_with_clip(img, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None, clip=True):
    """Resize image & clip pixel value over shooting 
            as a result of using InterpolationMode.{BICUBIC, LANCZOS}
       Similar to how scikit-image handles resizing
           https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize

        """
    if (not clip) or (interpolation not in [InterpolationMode.BICUBIC, InterpolationMode.LANCZOS]):
        return resize(img, size, interpolation=interpolation, max_size=max_size, antialias=antialias)
        
    if isinstance(img, torch.Tensor):
        min_val, max_val = img.min().item(), img.max().item()
    if isinstance(img, Image.Image):
        min_val, max_val = img.getextrema()
        
    img = resize(img, size, interpolation=interpolation, max_size=max_size, antialias=antialias)
        
    if isinstance(img, torch.Tensor):
        img = torch.clamp(img, min_val, max_val)
    if isinstance(img, Image.Image):
        img = np.array(img).clip(min_val, max_val)
        img = Image.fromarray(img)
        
    return img
    
    
class ResizeWithClip(Resize):
    """https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#Resize """
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None, clip=True):
        super().__init__(size, interpolation=interpolation, max_size=max_size, antialias=antialias)
        self.clip = clip
        
    def forward(self, img):
        return resize_with_clip(img, self.size, self.interpolation, self.max_size, self.antialias, self.clip)

    def __repr__(self) -> str:
        detail = (f"(size={self.size}, interpolation={self.interpolation.value}, "
                  f"max_size={self.max_size}, antialias={self.antialias}) clip={self.clip}")
        return f"{self.__class__.__name__}{detail}"


def _blend(img1, img2, ratio):
    """ Alpha blending """
    ratio = float(ratio)
    bound = 1.0 if img1.is_floating_point() else 255.0
    return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)


def adjust_brightness(img, brightness_factor):
    """Adjust brightness for grayscale tensor
    https://github.com/pytorch/vision/blob/36daee3f8f0d56eb869d7d5c2c4362bf1dc9a394/torchvision/transforms/functional_tensor.py#L258 """
    return _blend(img, torch.zeros_like(img), brightness_factor)


def adjust_contrast(im, contrast_factor):
    """Adjust contrast for grayscale tensor
    https://github.com/pytorch/vision/blob/e0467c64e337c0d1140a9f9a70a413b7268231f4/torchvision/transforms/functional_tensor.py#L176 """
    dtype = im.dtype if torch.is_floating_point(im) else torch.float32
    mean = torch.mean(im.to(dtype), dim=(-3, -2, -1), keepdim=True)
    return _blend(im, mean, contrast_factor)


class GrayscaleJitter(object):
    # Randomly change the brightness, contrast of grayscale image.
    # Reference:
    # https://github.com/pytorch/vision/blob/36daee3f8f/torchvision/transforms/transforms.py#L1007

    def __init__(self, brightness=0, contrast=0):
        super().__init__()

        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')

    def __call__(self, im):
        # Do randomized brightness/contrast adjustment to image, in a randomized order
        #
        fn_idx = torch.randperm(2)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                im = adjust_brightness(im, torch.tensor(1.0).uniform_(
                    self.brightness[0], self.brightness[1]).item())
            if fn_id == 1 and self.contrast is not None:
                im = adjust_contrast(im, torch.tensor(
                    1.0).uniform_(self.contrast[0], self.contrast[1]).item())
        return im

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, float) or isinstance(value, int):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(
                    "{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast
        if value[0] == value[1] == center:
            value = None
        return value
