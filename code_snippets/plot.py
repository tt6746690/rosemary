import numpy as np
import matplotlib.pyplot as plt


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
    cax = divider.append_axes("right", size="5%", pad=0.05)
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

