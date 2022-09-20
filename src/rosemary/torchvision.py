from typing import List, Tuple
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
    def forward(self, feature_maps: List[Tensor], image_size: Tuple[int]) -> Tensor:
        """Generate `anchors` given `feature_maps` computed from images with fixed `image_size`. """
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        self.set_cell_anchors(dtype, device)
        strides = [
            [torch.empty((), dtype=torch.int64, device=device).fill_(image_size[0] // g[0]),
             torch.empty((), dtype=torch.int64, device=device).fill_(image_size[1] // g[1]),]
                for g in grid_sizes]
        anchors = self.grid_anchors(grid_sizes, strides)
        anchors = torch.cat(anchors)
        return anchors