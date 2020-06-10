import os

DATA_DIR = 'Data'
STL_DIR = os.path.join(DATA_DIR, 'stl')
PROJ_DIR = os.path.join(DATA_DIR, 'projection_1024')
SPLIT_DIR = os.path.join(DATA_DIR, 'split')
VOXEL_TXT_DIR = os.path.join(DATA_DIR, 'voxel_txt_256')
VOXEL_NPY_DIR = os.path.join(DATA_DIR, 'voxel_npy_256')


PROJ_LEN = 1024
PROJ_SHAPE = (PROJ_LEN, PROJ_LEN)
SPLIT_PROJ_LEN = 64
SPLIT_PROJ_SHAPE = (SPLIT_PROJ_LEN, SPLIT_PROJ_LEN)
CAMERA_DIRS = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
NUM_VIEWS = len(CAMERA_DIRS)
VOXEL_LEN = 256
VOXEL_SHAPE = (VOXEL_LEN, VOXEL_LEN, VOXEL_LEN)
SPLIT_VOX_LEN = 16

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]