from config import *
import sentencepiece as spm
import csv


def read_split(in_path):
    """read the data and split it into source and target sequences

    Args:
        in_path (str): path to the input file

    Returns:
        list: a list of tuples containing the source and target sequences
    """
    pairs = []

    with open(in_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        for row in reader:
            src, tgt = row
            pairs.append((src, tgt))

    return pairs


if __name__ == "__main__":
    train_pairs = read_split(TRAIN_DATA)

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    with open(TOKENIZER_DIR / "sp_corpus.txt", "w", encoding="utf-8") as f:
        for src, tgt in train_pairs:
            f.write(src + "\n" + tgt + "\n")

    spm.SentencePieceTrainer.train(
        input=str(TOKENIZER_DIR / "sp_corpus.txt"),
        model_prefix=str(TOKENIZER_DIR / MODEL_PREFIX),
        vocab_size=VOCAB_SIZE,
        model_type="unigram",
        character_coverage=1.0,
        user_defined_symbols=[ANS_OPEN, ANS_CLOSE],
        pad_id=PAD_IDX,
        unk_id=UNK_IDX,
        bos_id=SOS_IDX,
        eos_id=EOS_IDX,
    )

    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "5_tokenizer_results.txt", "w", encoding="utf-8") as f:
        for src, _ in train_pairs[:5]:
            f.write(f"original: {src}\ntokenized: {sp.encode(src, out_type=str)}\n\n")
