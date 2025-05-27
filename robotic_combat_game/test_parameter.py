EMBEDDING_DIM = 128
K_SIZE = 9  # the number of neighbors

USE_GPU = False  # do you want to use GPUS?
NUM_GPU = 1  # the number of GPUs
NUM_META_AGENT = 10  # the number of processes

NUM_TEST = 10
NUM_RUN = 1
SAVE_GIFS = 0  # do you want to save GIFs
SAVE_TRAJECTORY = 0  # do you want to save per-step metrics
SAVE_LENGTH = 0  # do you want to save per-episode metrics
N_SELF = 3
N_ENEMY = 3
MAX_BLOOD = 10
INPUT_DIM = N_ENEMY*2+N_SELF*2+2+2
RANDOM_SEED = None

extra_info = ''
FOLDER_NAME = '{}V{}-{}'.format(
    N_SELF,
    N_ENEMY,
    extra_info
)
model_path = f'model/{FOLDER_NAME}'
gifs_path = f'results/{FOLDER_NAME}/gifs'
trajectory_path = f'results/{FOLDER_NAME}/trajectory/'
