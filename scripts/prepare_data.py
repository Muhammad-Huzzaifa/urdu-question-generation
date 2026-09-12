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
            tgt = " ".join(example["question"].split())

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
    wiki = load_dataset(WIKI_DATA_NAME)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_pairs = build_split(ds["train"], TRAIN_DATA)
    valid_pairs = build_split(ds["validation"], VALID_DATA)
    wiki_pairs = build_split(wiki["train"], WIKI_DATA)

    def mean_len(pairs, idx):
        """calculate the mean length of src / target pairs selected by idx

        Args:
            pairs (list): list of pairs containing src and tgt sentences
            idx (int): idx for choosing src (0) or the target (1)

        Returns:
            float: mean length of source / target sentences
        """
        return sum(len(p[idx].split()) for p in pairs) / len(pairs)

    rows = [
        (
            "Rows in raw dataset", 
            len(ds["train"]), len(ds["validation"]), len(wiki["train"])
        ),
        (
            "Answerable rows", 
            len(ds["train"]) - sum(ds["train"]["is_impossible"]), 
            len(ds["validation"]) - sum(ds["validation"]["is_impossible"]),
            len(wiki["train"]) - sum(wiki["train"]["is_impossible"])
        ),
        (
            "Pairs after length filter", 
            len(train_pairs), len(valid_pairs), len(wiki_pairs)),
        (
            "Mean source / target length",
            f"{mean_len(train_pairs, 0):.1f} / {mean_len(train_pairs, 1):.1f}",
            f"{mean_len(valid_pairs, 0):.1f} / {mean_len(valid_pairs, 1):.1f}",
            f"{mean_len(wiki_pairs, 0):.1f} / {mean_len(wiki_pairs, 1):.1f}",
        ),
    ]
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "dataset_statistics.md", "w", encoding="utf-8") as f:
        f.write("| | Train | Validation | Wiki-UQA |\n")
        f.write("|---|---|---|---|\n")
        for label, train_val, valid_val, wiki_val in rows:
            f.write(f"| {label} | {train_val} | {valid_val} | {wiki_val} |\n")

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    datasets = [
        ("train", train_pairs),
        ("valid", valid_pairs),
        ("wiki", wiki_pairs),
    ]

    fig, axes = plt.subplots(len(datasets), 2, figsize=(10, 4 * len(datasets)))

    for row, (name, pairs) in enumerate(datasets):
        src_lens = [len(src.split()) for src, _ in pairs]
        tgt_lens = [len(tgt.split()) for _, tgt in pairs]

        axes[row, 0].hist(src_lens, bins=30)
        axes[row, 0].set_title(f"distribution of source lengths in {name}")
        axes[row, 0].set_xlabel("number of words")
        axes[row, 0].set_ylabel("count")

        axes[row, 1].hist(tgt_lens, bins=30)
        axes[row, 1].set_title(f"distribution of target lengths in {name}")
        axes[row, 1].set_xlabel("number of words")
        axes[row, 1].set_ylabel("count")

    fig.tight_layout()
    fig.savefig(FIGS_DIR / "length_histograms.png")
