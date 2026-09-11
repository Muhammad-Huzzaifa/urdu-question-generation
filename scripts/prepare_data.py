from config import *
from datasets import load_dataset
import csv
import matplotlib.pyplot as plt


def split_sentences(text):
    """take the paragraph and extract complete sentences from it

    Args:
        text (str): context paragraph

    Yields:
        tuple: a tuple containing the start index, end index, and the extracted sentence
    """
    start = 0
    for i, ch in enumerate(text):

        if ch in SENT_DELIMS:
            yield start, i + 1, text[start : i + 1]
            start = i + 1

    if start < len(text):
        yield start, len(text), text[start:]


def make_pair(example, max_src=MAX_SOURCE_LENGTH, max_tgt=MAX_TARGET_LENGTH):
    """make pairs that will be used for training
    src is a sentence with <ans>...</ans> marked
    tgt is the question entry of example

    Args:
        example (dict): a single example from the dataset
        max_src (int, optional): maximum length for the source sequence, defaults to MAX_SOURCE_LENGTH
        max_tgt (int, optional): maximum length for the target sequence, defaults to MAX_TARGET_LENGTH

    Returns:
        tuple: a tuple containing the source and target sequences, or None if the example is not suitable for training
    """
    if example["is_impossible"]:
        return None
    
    a_start = example["answer_start"]
    a_text = example["answer"]
    context = example["context"]

    for s, e, sent in split_sentences(context):
        if s <= a_start < e:

            rel = a_start - s
            if sent[rel : rel + len(a_text)] != a_text:
                return None
            
            src = (sent[:rel] + " " + ANS_OPEN + " " + a_text + " " + ANS_CLOSE + " " + sent[rel + len(a_text) :]).strip()
            src = " ".join(src.split())
            tgt = SOS_TOKEN + " " + " ".join(example["question"].split()) + " " + EOS_TOKEN

            if len(src.split()) > max_src or len(tgt.split()) > max_tgt:
                return None
            
            return src, tgt
        
    return None


def build_split(split, out_path):
    """make pairs of the split(train/valid) and store it

    Args:
        split (Dataset): split passed that need to be processed and stored
        out_path (Path): output path where the processed pairs will be stored
    """
    pairs = [p for p in map(make_pair, split) if p is not None]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerows(pairs)
    print(f"{out_path}: {len(pairs)} pairs")

    return pairs


if __name__ == "__main__":
    ds = load_dataset(DATASET_NAME)
    print(ds)

    ex = ds["train"][0]
    print(ex.keys())
    print(ex["question"])
    print(ex["answer"])

    n_total = len(ds["train"])
    n_ans = n_total - sum(ds["train"]["is_impossible"])
    print (f"train rows: {n_total}, answerable: {n_ans}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_pairs = build_split(ds["train"], TRAIN_DATA)
    valid_pairs = build_split(ds["validation"], VALID_DATA)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "data_counts.txt", "w", encoding="utf-8") as f:
        f.write(f"train: \n\ttotal: {len(ds['train'])} \n\tanswerable: {len(ds['train']) - sum(ds['train']['is_impossible'])} \n\tselected: {len(train_pairs)}\n")
        f.write(f"validation: \n\ttotal: {len(ds['validation'])} \n\tanswerable: {len(ds['validation']) - sum(ds['validation']['is_impossible'])} \n\tselected: {len(valid_pairs)}\n")

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist([len(src.split()) for src, _ in train_pairs], bins=30, label="train source")
    plt.title("distribution of source lengths in train")
    plt.xlabel("number of words")
    plt.ylabel("count")
    plt.savefig(FIGS_DIR / "train_source_lengths.png")

    plt.figure()
    plt.hist([len(tgt.split()) for _, tgt in train_pairs], bins=30, label="train target")
    plt.title("distribution of target lengths in train")
    plt.xlabel("number of words")
    plt.ylabel("count")
    plt.savefig(FIGS_DIR / "train_target_lengths.png")

    plt.figure()
    plt.hist([len(src.split()) for src, _ in valid_pairs], bins=30, label="valid source")
    plt.title("distribution of source lengths in valid")
    plt.xlabel("number of words")
    plt.ylabel("count")
    plt.savefig(FIGS_DIR / "valid_source_lengths.png")

    plt.figure()
    plt.hist([len(tgt.split()) for _, tgt in valid_pairs], bins=30, label="valid target")
    plt.title("distribution of target lengths in valid")
    plt.xlabel("number of words")
    plt.ylabel("count")
    plt.savefig(FIGS_DIR / "valid_target_lengths.png")
