from config import *
import sacrebleu
from rouge_score import rouge_scorer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import math
import sentencepiece as spm
from scripts.train_tokenizer import read_split
from scripts.train import UQADataset, collate_fn
from app.model import Seq2Seq
import csv
import matplotlib.pyplot as plt


class WhitespaceTokenizer:
    def tokenize(self, text):
        return text.split()

def score(hyps, refs):
    """calculate BLEU-4, ROUGE-L, and <unk>% for a list of hypotheses/references.

    Args:
        hyps (list): a list of hypotheses
        refs (list): a list of references

    Returns:
        tuple: a tuple containing BLEU-4, ROUGE-L, and <unk>% scores
    """
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=WhitespaceTokenizer())
    rl = sum(scorer.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)) / len(refs)
    unk_rate = sum(h.count("\u2047") for h in hyps) / max(1, sum(len(h.split()) for h in hyps))
    return bleu, rl, unk_rate


def decode_split(model, dataloader, sp, beam_size, max_len):
    """decode a split of data using the model.decode()

    Args:
        model (Seq2Seq): the sequence-to-sequence model
        dataloader (DataLoader): the data loader for the split
        sp (SentencePieceProcessor): the sentencepiece processor for decoding
        beam_size (int): the beam size to use for decoding
        max_len (int): the maximum length of the decoded sequence

    Returns:
        tuple: a tuple containing the source sentences, reference sentences, and hypothesis sentences
    """
    model.eval()
    srcs, refs, hyps = [], [], []
 
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(DEVICE)
            lengths = (src != PAD_IDX).sum(dim=0)
            decoded_tokens, _ = model.decode(src, lengths, max_len=max_len, beam_size=beam_size)
 
            for i in range(len(decoded_tokens)):
                cnt = sp.decode([t for t in src[:, i].tolist() if t != PAD_IDX])
                hyp = sp.decode(decoded_tokens[i])
                ref = sp.decode([t for t in tgt[:, i].tolist() if t not in [PAD_IDX, SOS_IDX, EOS_IDX]])
 
                srcs.append(cnt)
                refs.append(ref)
                hyps.append(hyp)
 
    return srcs, refs, hyps


def compute_perplexity(model, dataloader, pad_idx):
    """compute the perplexity of the model on a given dataloader

    Args:
        model (Seq2Seq): the sequence-to-sequence model
        dataloader (DataLoader): the data loader for the split
        pad_idx (int): the index of the padding token

    Returns:
        float: the perplexity of the model on the given dataloader
    """
    model.eval()
    criterion_sum = CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
 
    total_loss, total_tokens = 0.0, 0
 
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            lengths = (src != pad_idx).sum(dim=0)
 
            output = model(src, lengths, tgt, teacher_forcing_ratio=1.0)
            target = tgt[1:].reshape(-1)
 
            loss = criterion_sum(output.reshape(-1, output.size(-1)), target)
            n_tokens = (target != pad_idx).sum().item()
 
            total_loss += loss.item()
            total_tokens += n_tokens
 
    return math.exp(total_loss / total_tokens)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))

    valid_pairs = read_split(VALID_DATA)
    valid_pairs = [(sp.encode(s), sp.encode(t, add_bos=True, add_eos=True)) for s, t in valid_pairs]
    valid_dataset = UQADataset(valid_pairs)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: collate_fn(x, PAD_IDX)
    )

    wiki_pairs = read_split(WIKI_DATA)
    wiki_pairs = [(sp.encode(s), sp.encode(t, add_bos=True, add_eos=True)) for s, t in wiki_pairs]
    wiki_dataset = UQADataset(wiki_pairs)
    wiki_dataloader = DataLoader(
        wiki_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: collate_fn(x, PAD_IDX)
    )

    model = Seq2Seq(
        device=DEVICE,
        pad_idx=PAD_IDX,
        sos_idx=SOS_IDX,
        eos_idx=EOS_IDX,
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
    model.eval()

    splits = [
        ("UQA valid", valid_dataloader),
        ("Wiki-UQA", wiki_dataloader),
    ]

    rows = []

    valid_srcs_greedy, valid_refs_greedy, valid_hyps_greedy = decode_split(model, valid_dataloader, sp, beam_size=1)
    valid_srcs_beam, valid_refs_beam, valid_hyps_beam = decode_split(model, valid_dataloader, sp, beam_size=BEAM_K)

    for split_name, dataloader in splits:
        ppl = compute_perplexity(model, dataloader, PAD_IDX)
 
        if split_name == "UQA valid":
            greedy_hyps, greedy_refs = valid_hyps_greedy, valid_refs_greedy
            beam_hyps, beam_refs = valid_hyps_beam, valid_refs_beam
        else:
            _, greedy_refs, greedy_hyps = decode_split(model, dataloader, sp, beam_size=1)
            _, beam_refs, beam_hyps = decode_split(model, dataloader, sp, beam_size=BEAM_K)
 
        bleu_g, rl_g, unk_g = score(greedy_hyps, greedy_refs)
        bleu_b, rl_b, unk_b = score(beam_hyps, beam_refs)
 
        rows.append((split_name, "greedy", bleu_g, rl_g, ppl, unk_g * 100))
        rows.append((split_name, f"beam (k={BEAM_K})", bleu_b, rl_b, ppl, unk_b * 100))

    with open(RESULTS_DIR / "automatic_metrics.md", "w", encoding="utf-8") as f:
        f.write("| Split | Decoding | BLEU-4 | ROUGE-L | PPL | <unk> % |\n")
        f.write("|---|---|---|---|---|---|\n")
        for split_name, decoding, bleu, rl, ppl, unk_pct in rows:
            f.write(f"| {split_name} | {decoding} | {bleu:.2f} | {rl:.2f} | {ppl:.2f} | {unk_pct:.2f} |\n")
 
    with open(RESULTS_DIR / "samples.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(["source", "reference", "greedy", f"beam_k{BEAM_K}"])
        for i in range(50):
            writer.writerow([
                valid_srcs_greedy[i],
                valid_refs_greedy[i],
                valid_hyps_greedy[i],
                valid_hyps_beam[i],
            ])

    with open(RESULTS_DIR / "human_evaluation.md", "w", encoding="utf-8") as f:
        f.write("| | Fluency | Relevance | Answerability |\n")
        f.write("|---|---|---|---|\n")
        f.write("| Member 1 (% yes) | | | |\n")
        f.write("| Member 2 (% yes) | | | |\n")
        f.write("| Cohen's kappa | | | |\n")

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    src0, _ = next(iter(valid_dataloader))
    src0 = src0[:, 0:1].to(DEVICE)
    lengths0 = (src0 != PAD_IDX).sum(dim=0)
    decoded_tokens0, attentions0 = model.decode(src0, lengths0, max_len=MAX_TARGET_LENGTH, beam_size=BEAM_K)
 
    src_tokens0 = [sp.id_to_piece(t) for t in src0[:, 0].tolist() if t != PAD_IDX]
    gen_tokens0 = [sp.id_to_piece(t) for t in decoded_tokens0[0]]
    attn_matrix0 = attentions0[0].numpy()
 
    fig, ax = plt.subplots(figsize=(max(6, len(src_tokens0) * 0.5), max(4, len(gen_tokens0) * 0.5)))
    im = ax.imshow(attn_matrix0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(src_tokens0)))
    ax.set_xticklabels(src_tokens0, rotation=90)
    ax.set_yticks(range(len(gen_tokens0)))
    ax.set_yticklabels(gen_tokens0)
    ax.set_xlabel("Source tokens")
    ax.set_ylabel("Generated tokens")
    ax.set_title("Attention weights")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "attention_heatmap.png")
    plt.close(fig)
