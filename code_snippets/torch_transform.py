import torch

def _blend(img1, img2, ratio):
    # Alpha blending
    # https://github.com/pytorch/vision/blob/36daee3f8f0d56eb869d7d5c2c4362bf1dc9a394/torchvision/transforms/functional_tensor.py#L557
    bound = 1.0 if img1.is_floating_point() else 255.0
    return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)


def adjust_brightness(im, brightness_factor):
    # Adjust brightness for grayscale tensor
    # https://github.com/pytorch/vision/blob/36daee3f8f0d56eb869d7d5c2c4362bf1dc9a394/torchvision/transforms/functional_tensor.py#L258
    return _blend(im, torch.zeros_like(im), brightness_factor)


def adjust_contrast(im, contrast_factor):
    # Adjust contrast for grayscale tensor
    # https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional_tensor.py#L258
    return _blend(im, im, contrast_factor)


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