from pathlib import Path

import torch


# dirs
ROOT_DIR = Path(__file__).parent

DATA_DIR = ROOT_DIR / "data"
TRAIN_DATA = DATA_DIR / "train.tsv"
VALID_DATA = DATA_DIR / "valid.tsv"

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

MAX_SOURCE_LENGTH = 60
MAX_TARGET_LENGTH = 25

ANS_OPEN = "<ans>"
ANS_CLOSE = "</ans>"
SENT_DELIMS = "\u06D4\u061F!"


# model
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.3
TEACHER_FORCING_RATIO = 0.5


# tokenizer
VOCAB_SIZE = 8000

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


# device
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
