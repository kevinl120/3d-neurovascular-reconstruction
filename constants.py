import os

DATA_DIR = 'Data'
STL_DIR = os.path.join(DATA_DIR, 'stl')
PROJ_DIR = os.path.join(DATA_DIR, 'projection_256')
VOXEL_TXT_DIR = os.path.join(DATA_DIR, 'voxel_txt')
VOXEL_NPY_DIR = os.path.join(DATA_DIR, 'voxel_npy_64')

PROJ_LEN = 256
PROJ_SHAPE = (PROJ_LEN, PROJ_LEN)
CAMERA_DIRS = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
NUM_VIEWS = len(CAMERA_DIRS)
VOXEL_LEN = 64
VOXEL_SHAPE = (VOXEL_LEN, VOXEL_LEN, VOXEL_LEN)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]