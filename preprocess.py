import cv2
import glob
import numpy as np
import os
from stl.mesh import Mesh
import vtkplotlib as vpl

from constants import *

def rename_data(dirs=[STL_DIR, VOXEL_TXT_DIR], names=['morph', 'voxel']):
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


def get_projections(file, camera_directions=CAMERA_DIRS):
    """ Get projection views in given camera directions from an STL file.

    Returns: Numpy array of shape (PROJ_LEN, PROJ_LEN, len(camera_directions)).
    """
    views = []
    for dir in camera_directions:
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
    views = np.array(views).reshape((PROJ_LEN, PROJ_LEN, len(camera_directions)))
    return views


def get_proj_files(stl_dir=STL_DIR, proj_dir=PROJ_DIR, camera_directions=CAMERA_DIRS):
    """ Write projection(s) of each file in stl_dir into proj_dir.
    
    Note: My device crashes when running more than 40 files continously. """
    stl_files = glob.glob(os.path.join(stl_dir, '*.stl'))
    stl_files.sort()
    for f in stl_files:
        views = get_projections(f, camera_directions=camera_directions)
        fname = f.split('/')[-1].split('.')[0] + '.npy'
        np.save(os.path.join(proj_dir, fname), views)


def convert_voxel_files(voxel_txt_dir=VOXEL_TXT_DIR, voxel_npy_dir=VOXEL_NPY_DIR):
    """ Convert coordinate triplet files into 3D matrix files. """
    voxel_files = glob.glob(os.path.join(voxel_txt_dir, '*.txt'))
    voxel_files.sort()
    for f in voxel_files[:1]:
        arr = np.zeros(VOXEL_SHAPE)
        voxels = np.genfromtxt(f, dtype=int, delimiter=', ')

        # Center voxelized construction
        x_shift = (VOXEL_LEN - np.max(voxels[:, 0])) // 2
        y_shift = (VOXEL_LEN - np.max(voxels[:, 1])) // 2
        z_shift = (VOXEL_LEN - np.max(voxels[:, 2])) // 2

        for v in voxels:
            arr[v[0]+x_shift][v[1]+y_shift][v[2]+z_shift] = 1
        fname = f.split('/')[-1].split('.')[0] + '.npy'
        np.save(os.path.join(voxel_npy_dir, fname), arr)


def load_data(proj_dir=PROJ_DIR, vox_dir=VOXEL_NPY_DIR):
    proj_files = glob.glob(os.path.join(proj_dir, '*.npy'))
    proj_files.sort()
    x = []
    for f in proj_files:
        x.append(np.load(f))
    voxel_files = glob.glob(os.path.join(vox_dir, '*.npy'))
    voxel_files.sort()
    y = []
    for f in voxel_files:
        y.append(np.load(f))
    x = np.array(x)
    y = np.array(y)
    print("Loaded data with shapes x: {}, y: {}".format(x.shape, y.shape))
    return x, y


def split_data(proj_dir=PROJ_DIR, vox_dir=VOXEL_NPY_DIR, split_dir=SPLIT_DIR):
    # hardcoded for the specific order of 6 camera views
    proj, vox = load_data(proj_dir, vox_dir)

    p = proj[0]
    p = p.reshape((-1, PROJ_LEN, PROJ_LEN))
    v = vox[0]

    x = []
    y = []

    # for i in range(16):
    #     for j in range(16):
    #         for k in range(16):

    i = 8
    j = 8
    k = 8

    # i = 13
    # j = 6
    # k = 6

    projs = []
    view_0 = p[0, i*SPLIT_PROJ_LEN:(i+1)*SPLIT_PROJ_LEN, j*SPLIT_PROJ_LEN:(j+1)*SPLIT_PROJ_LEN]
    view_1 = p[1, (i)*SPLIT_PROJ_LEN:(i+1)*SPLIT_PROJ_LEN, PROJ_LEN-(j+1)*SPLIT_PROJ_LEN:PROJ_LEN-(j)*SPLIT_PROJ_LEN]
    view_2 = p[2, PROJ_LEN-(k+1)*SPLIT_PROJ_LEN:PROJ_LEN-(k)*SPLIT_PROJ_LEN, j*SPLIT_PROJ_LEN:(j+1)*SPLIT_PROJ_LEN]
    view_3 = p[3, PROJ_LEN-(k+1)*SPLIT_PROJ_LEN:PROJ_LEN-(k)*SPLIT_PROJ_LEN, PROJ_LEN-(j+1)*SPLIT_PROJ_LEN:PROJ_LEN-(j)*SPLIT_PROJ_LEN]
    view_4 = p[4, i*SPLIT_PROJ_LEN:(i+1)*SPLIT_PROJ_LEN, PROJ_LEN-(k+1)*SPLIT_PROJ_LEN:PROJ_LEN-(k)*SPLIT_PROJ_LEN] 
    view_5 = p[5, i*SPLIT_PROJ_LEN:(i+1)*SPLIT_PROJ_LEN, k*SPLIT_PROJ_LEN:(k+1)*SPLIT_PROJ_LEN]
    projs.append(view_0)
    projs.append(view_1)
    projs.append(view_2)
    projs.append(view_3)
    projs.append(view_4)
    projs.append(view_5)
    projs = np.array(projs)

    # projs = []
    # for i in range(16):
    #     for k in range(16):
    #         view_0 = p[4, i*SPLIT_PROJ_LEN:(i+1)*SPLIT_PROJ_LEN, PROJ_LEN-(k+1)*SPLIT_PROJ_LEN:PROJ_LEN-(k)*SPLIT_PROJ_LEN] 
    #         projs.append(view_0)
    # projs = np.array(projs)

    # projs = projs.reshape(SPLIT_PROJ_LEN, SPLIT_PROJ_LEN, 6)

    y = v[(j)*SPLIT_VOX_LEN:(j+1)*SPLIT_VOX_LEN, (k)*SPLIT_VOX_LEN:(k+1)*SPLIT_VOX_LEN, VOXEL_LEN-(i+1)*SPLIT_VOX_LEN:VOXEL_LEN-(i)*SPLIT_VOX_LEN]

    return projs, y


def make_data_dirs():
    os.makedirs(STL_DIR, exist_ok=True)
    os.makedirs(PROJ_DIR, exist_ok=True)
    os.makedirs(VOXEL_TXT_DIR, exist_ok=True)
    os.makedirs(VOXEL_NPY_DIR, exist_ok=True)


def main():
    # make_data_dirs()
    # rename_data(dirs=[VOXEL_TXT_DIR], names=['voxel'])
    # get_proj_files()
    convert_voxel_files()

    # split_proj()
    pass


if __name__ == '__main__':
    main()