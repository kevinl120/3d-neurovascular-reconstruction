import os

DATA_DIR = 'Data'
STL_DIR = os.path.join(DATA_DIR, 'stl')
PROJ_DIR = os.path.join(DATA_DIR, 'projection')
VOXEL_TXT_DIR = os.path.join(DATA_DIR, 'voxel_txt')
VOXEL_NPY_DIR = os.path.join(DATA_DIR, 'voxel_npy')

PROJ_SHAPE = (256, 256)
VOXEL_SHAPE = (64, 64, 64)