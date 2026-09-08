from pathlib import Path


# dirs
ROOT_DIR = Path(__file__).parent

DATA_DIR = ROOT_DIR / "data"
TRAIN_DATA = DATA_DIR / "train.tsv"
VALID_DATA = DATA_DIR / "valid.tsv"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
TOKENIZER_DIR = ARTIFACTS_DIR / "tokenizer"
MODEL_PREFIX = "ur_sp"
TOKENIZER_MODEL = TOKENIZER_DIR / f"{MODEL_PREFIX}.model"

RESULTS_DIR = ROOT_DIR / "results"
FIGS_DIR = RESULTS_DIR / "figures"


# data & prep
DATASET_NAME = "uqa/UQA"

MAX_SOURCE_LENGTH = 60
MAX_TARGET_LENGTH = 25

ANS_OPEN = "<ans>"
ANS_CLOSE = "</ans>"
SENT_DELIMS = "\u06D4\u061F!"


# tokenizer
VOCAB_SIZE = 8000
