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
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )

    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    src, tgt = train_pairs[0]
    print(sp.encode(src, out_type=str))
    print(sp.encode(tgt))
    print(sp.decode(sp.encode(tgt)) == tgt)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "5_tokenizer_results.txt", "w", encoding="utf-8") as f:
        for src, _ in train_pairs[:5]:
            f.write(f"original: {src}\ntokenized: {sp.encode(src, out_type=str)}\n\n")
