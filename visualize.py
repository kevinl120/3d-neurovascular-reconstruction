import glob
import matplotlib.pyplot as plt
import numpy as np

from constants import *

def plot_projection(proj):
    fig, ax = plt.subplots()
    if isinstance(proj, str):
        fig.suptitle(proj)
        proj = np.genfromtxt(proj, dtype=int, delimiter=',')
    ax.pcolor(proj, cmap='Greys', vmin=0.0, vmax=1.0)
    ax.set_aspect('equal')
    plt.axis('off')
    plt.show()


def main():
    proj_files = glob.glob(os.path.join(PROJ_DIR, '*.txt'))
    proj_files.sort()
    plot_projection(proj_files[0])


if __name__ == '__main__':
    main()