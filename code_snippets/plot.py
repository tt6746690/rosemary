import numpy as np
import matplotlib.pyplot as plt

def plt_scaled_colobar_ax(ax):
    """ Create color bar
            fig.colorbar(im, cax=plt_scaled_colobar_ax(ax)) 
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return cax


def show_slices(vol, colorbar=True):
    """RAS coordinate: https://www.fieldtriptoolbox.org/faq/coordsys/
    
    (x,y,z)
    -x indicate left hemisphere
    -y indicate rear of brain
    -z indicate bottom of brain
    """
    center = np.array(vol.shape)//2

    slices = [vol[center[0],:,:],  # saggital: slice through x
              vol[:,center[1],:],  # coronal:  slice through y
              vol[:,:,center[2]]]  # axial:    slice through z

    fig, axs = plt.subplots(1, len(slices), figsize=(8*len(slices), 8))

    for i, s in enumerate(slices):
        ax = axs[i]
        im = ax.imshow(s.T, cmap='gray', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        if colorbar:
            fig.colorbar(im, cax=plt_scaled_colobar_ax(ax))

    return fig, axs


def normalize_dmoyer(vol):
    """ Normalize volume `vol` by clipping extreme values 
        such that intensities roughly lies in the [0, 1] range 
        `vol_max` and `vol_denom` different to preserve dynamic range
    """
    vol = np.copy(vol)
    vol_max = np.quantile(vol,0.995)
    vol_denom = np.quantile(vol,0.99)
    vol = np.clip(vol, 0, vol_max)
    vol /= vol_denom
    return vol