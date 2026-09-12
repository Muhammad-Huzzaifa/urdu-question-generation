from pathlib import Path
import torch

# dirs
ROOT_DIR = Path(__file__).parent

DATA_DIR = ROOT_DIR / "data"
TRAIN_DATA = DATA_DIR / "train.tsv"
VALID_DATA = DATA_DIR / "valid.tsv"
WIKI_DATA = DATA_DIR / "wiki.tsv"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
TOKENIZER_DIR = ARTIFACTS_DIR / "tokenizer"
MODEL_PREFIX = "ur_sp"
TOKENIZER_MODEL = TOKENIZER_DIR / f"{MODEL_PREFIX}.model"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
BEST_MODEL = CHECKPOINT_DIR / "best_model.pt"

RESULTS_DIR = ROOT_DIR / "results"
FIGS_DIR = RESULTS_DIR / "figures"

# data & prep
DATASET_NAME = "uqa/UQA"
WIKI_DATA_NAME = "uqa/Wiki-UQA"

MAX_SOURCE_LENGTH = 60
MAX_TARGET_LENGTH = 25

ANS_OPEN = "<ans>"
ANS_CLOSE = "</ans>"
PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN = "<pad>", "<unk>", "<s>", "</s>"
SENT_DELIMS = "\u06D4\u061F!"

# model
BATCH_SIZE = 64
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3
TF_START = 0.9
TF_END = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SCHEDULER_FACTOR = 0.5
SCHEDULE_PATIENCE = 1
SCHEDULE_COOLDOWN = 0
MIN_LR = 1e-6
CLIP = 5.0
NUM_EPOCHS = 15
BEAM_K = 5

def tf_ratio(epoch):
    """calculate the teacher forcing ratio for a given epoch

    Args:
        epoch (int): the current epoch

    Returns:
        float: the teacher forcing ratio for the given epoch
    """
    return TF_START + (TF_END - TF_START) * (epoch / NUM_EPOCHS)


# tokenizer
VOCAB_SIZE = 8000

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"