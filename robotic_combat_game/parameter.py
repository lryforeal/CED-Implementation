REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 128
EMBEDDING_DIM = 128
NODE_PADDING_SIZE = 101  # the number of nodes will be padded to this value
K_SIZE = 9  # the number of neighboring nodes

USE_GPU = True  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs

NUM_GPU = 2
NUM_META_AGENT = 16
LR = 1e-5
GAMMA = 0.99
DECAY_STEP = 256
SUMMARY_WINDOW = 1
LOAD_MODEL = False # do you want to load the model trained before

N_SELF = 3
N_ENEMY = 3
MAX_BLOOD = 10
BETA = 0.1

INPUT_DIM = N_ENEMY*2+N_SELF*2+2+2
train_mode = True
RANDOM_SEED = None

extra_info = ''
FOLDER_NAME = '{}V{}-{}'.format(
    N_SELF,
    N_ENEMY,
    extra_info
)
model_path = f'model/{FOLDER_NAME}'
base_policy_path = f'data/base_policy/base_policy.pth'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'

adj_path = f'data/adj_file'
