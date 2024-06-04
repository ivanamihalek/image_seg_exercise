
FILE_PATHS = 'Human-Segmentation-Dataset-master/train.csv'
DATA_DIR = 'content'

DEVICE = 'cuda'  # because we are using GPU
EPOCHS = 50
LR = 0.003
IMG_SIZE = 320
BATCH_SIZE = 4  # this is the biggest batch size for which my card has enough memory

ENCODER = 'timm-efficientnet-b0'
WEIGHTS = 'imagenet'

