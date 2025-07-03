# PATHS 

CSV_DATASET_PATH = "../dataset/train.csv"
MODEL_PATH = "../model.pt"
TOKENIZER_PATH = "../tokenizer.json"

# MODEL ARCHITECTURE

ROPE_MAX_LEN = 20000 # positions for sin and cos vectors to be precomputed
MODEL_MLP_HIDDEN_LAYERS = 2
MODEL_MLP_HIDDEN_LAYER_SIZE = 512
MODEL_SIZE = 512
MODEL_NUMBER_OF_HEADS = 25
MODEL_HEAD_DIM = 100
MODEL_NUMBER_OF_BLOCKS = 8

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
SAVE_FREQ = 10000
PRINT_FREQ = 100

# INFERENCE

MAX_TOKENS = 2000
OUTPUT_TYPE = "sample" # or "argmax"
START_PROMPT = "Hello world, I am "
