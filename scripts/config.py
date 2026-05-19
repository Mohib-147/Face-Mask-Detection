import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

MODEL1_PATH = os.path.join(MODELS_DIR, 'model1_mask_status.pkl')
MODEL2_PATH = os.path.join(MODELS_DIR, 'model2_mask_colour.pkl')
MODEL3_PATH = os.path.join(MODELS_DIR, 'model3_mask_type.pkl')

MODEL1_LABELS = ['no_mask', 'proper_mask', 'improper_mask']
MODEL2_LABELS = ['black', 'blue', 'green', 'red', 'white', 'yellow']
MODEL3_LABELS = ['cloth', 'medical', 'n95']

IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
FLATTENED_SIZE = 224 * 224 * 3
