import glob
import matplotlib.pyplot as plt
import numpy as np

from constants import *
import preprocess

def plot_projection(proj, view=None):
    """ Plot projections from a file path or a numpy array.

    Parameters:
        proj (numpy array or string): Must be shape PROJ_SHAPE or (n, PROJ_SHAPE).
        view (int): View to plot. Defaults to all views.
    """
    if isinstance(proj, str):
        proj = np.load(proj)
    proj = proj.reshape((-1, SPLIT_PROJ_LEN, SPLIT_PROJ_LEN))
    # proj = proj.reshape((-1, PROJ_LEN, PROJ_LEN))
    if view is not None or proj.shape[0] == 1:
        if view is not None:
            arr = proj[view]
        else:
            arr = proj
        fig = plt.figure()
        ax = fig.gca()
        ax.pcolormesh(arr, cmap='Greys', vmin=0.0, vmax=1.0)
        ax.set_aspect('equal')
        plt.axis('off')
        plt.show()
    else:
        fig, ax = plt.subplots(nrows=2, ncols=3)
        # fig, ax = plt.subplots(nrows=16, ncols=16)
        for i, row in enumerate(ax):
            for j, col in enumerate(row):
                arr = proj[i*len(ax[0])+j]
                col.pcolormesh(arr, cmap='Greys', vmin=0.0, vmax=1.0)
                col.invert_yaxis()
                col.invert_xaxis()
                col.set_aspect('equal')
                col.axis('off')
                # col.set_title('{}, {}'.format(i, j))
        plt.show()


def plot_voxels(vox):
    fig = plt.figure()
    if isinstance(vox, str):
        fig.suptitle(vox)
        arr = np.load(vox)
    else:
        arr = vox
    ax = fig.gca(projection='3d')
    ax.voxels(arr)
    plt.show()


def main():
    # proj_files = glob.glob(os.path.join(PROJ_DIR, '*.npy'))
    # proj_files.sort()
    # for i in range(1):
    #     plot_projection(proj_files[i])

    # plot_voxels('./Data/voxel_npy_256/voxel_0000.npy')

    projs, y = preprocess.split_data()
    plot_projection(projs)
    plot_voxels(y)

    # x, y = preprocess.load_data()
    # plot_projection(x[0], view=0)

    pass


if __name__ == '__main__':
    main()