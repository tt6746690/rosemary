import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    'plt_kernel_matrix_one',
    'plt_scaled_colobar_ax',
    'plt_savefig',
    'plt_slices',
    'plt_det',
]


def plt_kernel_matrix_one(fig, ax, K, title=None, n_ticks=5,
                          custom_ticks=True, vmin=None, vmax=None, annotate=False):
    im = ax.imshow(K, vmin=vmin, vmax=vmax)
    ax.set_title(title if title is not None else '')
    fig.colorbar(im, cax=plt_scaled_colobar_ax(ax))
    # custom ticks
    if custom_ticks:
        n = len(K)
        ticks = list(range(n))
        ticks_idx = np.rint(np.linspace(
            1, len(ticks), num=min(n_ticks,    len(ticks)))-1).astype(int)
        ticks = list(np.array(ticks)[ticks_idx])
        ax.set_xticks(np.linspace(0, n-1, len(ticks)))
        ax.set_yticks(np.linspace(0, n-1, len(ticks)))
        ax.set_xticklabels(ticks)
        ax.set_yticklabels(ticks)
    if annotate:
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                ax.annotate(f'{K[i,j]:.2f}', xy=(j, i),
                            horizontalalignment='center',
                            verticalalignment='center')
    return fig, ax


def plt_scaled_colobar_ax(ax):
    """ Create color bar
            fig.colorbar(im, cax=plt_scaled_colobar_ax(ax)) 
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7.5%", pad=0.05)
    return cax


def plt_savefig(fig, save_path):
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=100)


def plt_slices(vol, colorbar=True):
    """RAS coordinate: https://www.fieldtriptoolbox.org/faq/coordsys/

    (x,y,z)
    -x indicate left hemisphere
    -y indicate rear of brain
    -z indicate bottom of brain
    """
    center = np.array(vol.shape)//2

    slices = [vol[center[0], :, :],  # saggital: slice through x
              vol[:, center[1], :],  # coronal:  slice through y
              vol[:, :, center[2]]]  # axial:    slice through z

    fig, axs = plt.subplots(1, len(slices), figsize=(8*len(slices), 8))

    for i, s in enumerate(slices):
        ax = axs[i]
        im = ax.imshow(s.T, cmap='gray', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        if colorbar:
            fig.colorbar(im, cax=plt_scaled_colobar_ax(ax))

    return fig, axs


def plt_det(fig, ax, image, boxes, labels=None, masks=None, colors=None):
    """Draw bounding box over image.
    
        `boxes`    List<List[4]>
            in (xmin, ymin, xmax, ymax) format

        based on:
        https://github.com/shoumikchow/bbox-visualizer/blob/master/bbox_visualizer/bbox_visualizer.py
    """
    import torch
    import bbox_visualizer as bbv
    from PIL import Image
    from .torch import torch_tensor_to_ndarray, torch_get_dimensions
    from .np import np_trim_upper

    N = len(boxes)
    C, H, W = torch_get_dimensions(image)

    if isinstance(labels, (str, int)):
        labels = [labels]*N
    if isinstance(labels, torch.Tensor):
        labels = torch_tensor_to_ndarray(labels)
    if isinstance(boxes, torch.Tensor):
        boxes = torch_tensor_to_ndarray(boxes)
    if isinstance(masks, torch.Tensor):
        masks = torch_tensor_to_ndarray(masks)
    if isinstance(image, Image.Image):
        image = np.array(image) # (H, W, C)
    if isinstance(image, torch.Tensor):
        image = torch_tensor_to_ndarray(image) # (H, W) | (C, H, W)
        image = image.transpose((1,2,0)) # (H, W, C)

    boxes = boxes.astype(int)

    if isinstance(image, np.ndarray):
        if C == 1: # (H, W) -> (H, W, C)
            image = np_trim_upper(np.stack((image.squeeze(),)*3, axis=-1), α=.001)
        image = (255*image).astype(np.uint8)

    
    if colors is None:
        colors = ((np.random.random((N, 3))*0.6+0.4)*255)

    line_thickness = max(int(2*min(H, W)/256), 1)
    
    for i in range(N):
        c, bbox = colors[i], boxes[i]
        image = bbv.draw_rectangle(
            image, bbox, c, thickness=line_thickness, is_opaque=False, alpha=.5)
        if labels is not None and labels[i]:
            image = bbv_add_label(
                image, labels[i], bbox, draw_bg=True, text_bg_color=c, top=False)
    ax.imshow(image)
    if masks is not None:
        mask = masks
        mask = (mask - np.min(mask))/np.ptp(mask)
        mask = np.ma.masked_where(mask==0, mask).astype(np.uint8)
        im = ax.imshow(mask, cmap='jet', alpha=.4)
        fig.colorbar(im, cax=plt_scaled_colobar_ax(ax))


def bbv_add_label(img, label, bbox, draw_bg=True, text_bg_color=(255, 255, 255), 
              text_color=(0, 0, 0), top=True):
    """Modified from https://github.com/shoumikchow/bbox-visualizer/blob/master/bbox_visualizer/bbox_visualizer.py """
    import cv2
    from .torch import torch_get_dimensions
    H, W, _ = img.shape
    thickness = max(int(1*min(H, W)/256), 1)
    fontScale = max(.3*min(H, W)/256, .3)
    offset = 0

    text_width, text_height = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)[0]

    if top:
        label_bg = [bbox[0], bbox[1], bbox[0] + text_width, bbox[1] - text_height]
        if draw_bg:
            cv2.rectangle(img, (label_bg[0], label_bg[1]),
                          (label_bg[2] + offset, label_bg[3]), text_bg_color, -1)
        cv2.putText(img, label, (bbox[0] + offset, bbox[1] - offset),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale, text_color, thickness)

    else:
        label_bg = [bbox[0], bbox[1], bbox[0] + text_width, bbox[1] + text_height]
        if draw_bg:
            cv2.rectangle(img, (label_bg[0], label_bg[1]),
                          (label_bg[2] + offset, label_bg[3]), text_bg_color, -1)
        cv2.putText(img, label, (bbox[0] + offset, bbox[1] - offset + text_height),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale, text_color, thickness)

    return img

