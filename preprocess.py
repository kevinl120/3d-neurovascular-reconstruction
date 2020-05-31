import cv2
import glob
import numpy as np
import os
from stl.mesh import Mesh
import vtkplotlib as vpl

from constants import *

def rename_data(dirs=[STL_DIR], names=['morph']):
    """ Renames files in each dir to {name}_0000.ext. """
    if len(dirs) != len(names):
        raise ValueError('Expected {} names for renaming {} dirs, instead found {} names'.format(len(dirs), len(dirs), len(names)))
    for dir, name in zip(dirs, names):
        paths = glob.glob(os.path.join(dir, '*'))
        paths.sort()
        for i, path in enumerate(paths):
            ext = path.split('/')[-1].split('.')[-1]
            new_path = os.path.join(dir, '{}_{:04d}.{}'.format(name, i, ext))
            os.rename(path, new_path)


def get_projections(file, camera_directions=[(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]):
    """ Get projection views in given camera directions from an STL file.

    Returns: List of numpy arrays of shape PROJ_SHAPE.
    """
    views = []
    for dir in camera_directions[::-1]:
        mesh = Mesh.from_file(file)
        vpl.mesh_plot(mesh, color="black")
        r = vpl.view(camera_position=(0,0,0), camera_direction=dir)
        vpl.reset_camera()
        vpl.zoom_to_contents(padding=1)
        # Upscale first so downscale is more accurate
        arr = vpl.screenshot_fig(magnification=10, off_screen=True)
        # Change 3-channel RGB image to single channel binary matrix
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        arr[arr == 0] = 1
        arr[arr == 218] = 0
        arr = cv2.resize(arr, dsize=PROJ_SHAPE, interpolation=cv2.INTER_LINEAR)
        vpl.close()
        views.append(arr)
    return views


def get_proj_files(stl_dir=STL_DIR, proj_dir=PROJ_DIR, camera_directions=[(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]):
    """ Write projection(s) of each file in stl_dir into proj_dir.
    
    Note: My device crashes when running more than 40 files continously. """
    stl_files = glob.glob(os.path.join(stl_dir, '*.stl'))
    stl_files.sort()
    for f in stl_files:
        views = get_projections(f, camera_directions=camera_directions)
        for i, v in enumerate(views):
            fname = f.split('/')[-1].split('.')[0] + '_{}.txt'.format(i)
            np.savetxt(os.path.join(proj_dir, fname), v, fmt='%i', delimiter=',')


def make_data_dirs():
    os.makedirs(STL_DIR, exist_ok=True)
    os.makedirs(PROJ_DIR, exist_ok=True)
    os.makedirs(VOXEL_DIR, exist_ok=True)


def main():
    # make_data_dirs()
    # get_proj_files()
    pass


if __name__ == '__main__':
    main()