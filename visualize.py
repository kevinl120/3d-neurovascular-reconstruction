import glob
import matplotlib.pyplot as plt
import numpy as np

from constants import *

def plot_projection(proj):
    fig = plt.figure()
    if isinstance(proj, str):
        fig.suptitle(proj)
        proj = np.genfromtxt(proj, dtype=int, delimiter=',')
    ax = fig.gca()
    ax.pcolormesh(proj, cmap='Greys', vmin=0.0, vmax=1.0)
    ax.set_aspect('equal')
    plt.axis('off')
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
    # proj_files = glob.glob(os.path.join(PROJ_DIR, '*.txt'))
    # proj_files.sort()
    # for i in range(30):
    #     plot_projection(proj_files[i*6])

    # plot_voxels('./Data/voxel_npy/voxel_0000.npy')

    pass


if __name__ == '__main__':
    main()