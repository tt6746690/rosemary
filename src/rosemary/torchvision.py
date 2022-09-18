from typing import List
import torch
from torch import Tensor
from torchvision.models.detection.anchor_utils import AnchorGenerator

__all__ = [
    "AnchorGeneratorFixedImageSize",
]


class AnchorGeneratorFixedImageSize(AnchorGenerator):
    """`AnchorGenerator` where `images` have the same shape.
        - no need to convert to `ImageList` in `forward` 
        - just need to compute anchors for one image, since 
            the anchors is fixed given fixed image shapes.
    """
    def __init__(
        self, 
        image_size,
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    ):
        super().__init__(sizes, aspect_ratios)
        if isinstance(image_size, (int, float)):
            image_size = (int(image_size),)*2
        self.image_size = image_size
    
    def forward(self, feature_maps: List[Tensor]) -> Tensor:
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [
            [torch.empty((), dtype=torch.int64, device=device).fill_(self.image_size[0] // g[0]),
             torch.empty((), dtype=torch.int64, device=device).fill_(self.image_size[1] // g[1]),]
                for g in grid_sizes]
        anchors = self.grid_anchors(grid_sizes, strides)
        anchors = torch.cat(anchors)
        return anchors