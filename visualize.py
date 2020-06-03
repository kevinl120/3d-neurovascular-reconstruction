import glob
import matplotlib.pyplot as plt
import numpy as np

from constants import *

def plot_projection(proj, view=None):
    """ Plot projections from a file path or a numpy array.

    Parameters:
        proj (numpy array or string): Must be shape PROJ_SHAPE or (n, PROJ_SHAPE).
        view (int): View to plot. Defaults to all views.
    """
    if isinstance(proj, str):
        proj = np.load(proj)
    if view is not None or proj.shape == PROJ_SHAPE:
        fig = plt.figure()
        proj = proj[view]
        ax = fig.gca()
        ax.pcolormesh(proj, cmap='Greys', vmin=0.0, vmax=1.0)
        ax.set_aspect('equal')
        plt.axis('off')
        plt.show()
    else:
        fig, ax = plt.subplots(nrows=2, ncols=proj.shape[0]//2)
        for i, row in enumerate(ax):
            for j, col in enumerate(row):
                view = proj[i*len(ax[0])+j]
                col.pcolormesh(view, cmap='Greys', vmin=0.0, vmax=1.0)
                col.set_aspect('equal')
                col.axis('off')
        plt.show()


def plot_voxels(vox):
    fig = plt.figure()
    if isinstance(vox, str):
        fig.suptitle(vox)
        arr = np.load(vox)
    ax = fig.gca(projection='3d')
    ax.voxels(arr)
    plt.show()


def main():
    proj_files = glob.glob(os.path.join(PROJ_DIR, '*.npy'))
    proj_files.sort()
    for i in range(1):
        plot_projection(proj_files[i])

    # plot_voxels('./Data/voxel_npy/voxel_0000.npy')

    pass


if __name__ == '__main__':
    main()