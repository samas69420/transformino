# PATHS 

CSV_DATASET_PATH = "../dataset/train.csv"
MODEL_PATH = "../model.pth"
TOKENIZER_PATH = "../tokenizer.json"

# MODEL ARCHITECTURE

ROPE_MAX_LEN = 20000 # positions for sin and cos vectors to be precomputed
MODEL_MLP_HIDDEN_LAYERS = 2
MODEL_MLP_HIDDEN_LAYER_SIZE = 10
MODEL_SIZE = 100
MODEL_NUMBER_OF_HEADS = 3
MODEL_HEAD_DIM = 100
MODEL_NUMBER_OF_BLOCKS = 2

# TRAINING (general)

TEXT_COLUMN = "text" # column of the csv that contains the text used for training

# TRAINING (tokenizer)

DESIRED_VOCAB_LEN = 5000 
MIN_FREQUENCY = 5
CHUNK_SIZE = 10_000

# TRAINING (model)

BATCH_SIZE = 10
LEARNING_RATE = 0.0001
EPOCHS = 10
SAVE_FREQ = 50
PRINT_FREQ = 10

# INFERENCE

MAX_TOKENS = 2000
OUTPUT_TYPE = "sample" # or "argmax"
START_PROMPT = "Hello world, I am "
