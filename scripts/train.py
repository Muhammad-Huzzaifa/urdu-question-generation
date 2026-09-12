from config import *
from scripts.train_tokenizer import read_split
import sentencepiece as spm
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from app.model import Seq2Seq
import time
from datetime import timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt


class UQADataset(Dataset):

    def __init__(self, pairs):
        """initialize the dataset with pairs of source and target sequences

        Args:
            pairs (list): a list of tuples containing source and target sequences
        """
        self.pairs = pairs

    def __len__(self):
        """get the length of the dataset

        Returns:
            int: the number of pairs in the dataset
        """
        return len(self.pairs)

    def __getitem__(self, idx):
        """get a specific pair from the dataset

        Args:
            idx (int): the index of the pair to retrieve

        Returns:
            tuple: a tuple containing the source and target sequences
        """
        return self.pairs[idx]


def collate_fn(batch, pad_idx):
    """collate a batch of sequences

    Args:
        batch (list): a list of tuples containing source and target sequences
        pad_idx (int): the index to use for padding

    Returns:
        tuple: a tuple containing the padded source and target sequences
    """
    src_batch, tgt_batch = zip(*batch)

    src_batch = [torch.tensor(src) for src in src_batch]
    tgt_batch = [torch.tensor(tgt) for tgt in tgt_batch]

    src_batch = pad_sequence(src_batch, padding_value=pad_idx)
    tgt_batch = pad_sequence(tgt_batch, padding_value=pad_idx)

    return src_batch, tgt_batch


def train_step(dataloader, model, optimizer, criterion, device, pad_idx, tf_ratio, clip):
    """perform a training on one epoch

    Args:
        dataloader (DataLoader): the data loader for the training data
        model (Seq2Seq): the sequence-to-sequence model
        optimizer (torch.optim.Optimizer): the optimizer to use for training
        criterion (torch.nn.Module): the loss function to use
        device (torch.device): the device to use for training
        pad_idx (int): the index to use for padding
        tf_ratio (float): the teacher forcing ratio to use during training
        clip (float): the maximum norm for gradient clipping

    Returns:
        float: the average loss for the epoch
    """
    model.train()
    total_loss = 0

    tqdm_bar = tqdm(dataloader, desc="Training", leave=False)
    for idx, (src, tgt) in enumerate(tqdm_bar):
        src, tgt = src.to(device), tgt.to(device)
        lengths = (src != pad_idx).sum(dim=0)

        optimizer.zero_grad()
        output = model(src, lengths, tgt, teacher_forcing_ratio=tf_ratio)
        loss = criterion(output.reshape(-1, output.size(-1)), tgt[1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (idx + 1)

        tqdm_bar.set_postfix({"avg_loss": f"{avg_loss:.4f}", "current_loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def evaluation_step(dataloader, model, criterion, device, pad_idx):
    """perform an evaluation on one epoch

    Args:
        dataloader (DataLoader): the data loader for the evaluation data
        model (Seq2Seq): the sequence-to-sequence model
        criterion (torch.nn.Module): the loss function to use
        device (torch.device): the device to use for evaluation
        pad_idx (int): the index to use for padding

    Returns:
        float: the average loss for the epoch
    """
    model.eval()
    total_loss = 0

    tqdm_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for idx, (src, tgt) in enumerate(tqdm_bar):
            src, tgt = src.to(device), tgt.to(device)
            lengths = (src != pad_idx).sum(dim=0)

            output = model(src, lengths, tgt, teacher_forcing_ratio=0.0)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[1:].reshape(-1))

            total_loss += loss.item()
            avg_loss = total_loss / (idx + 1)

            tqdm_bar.set_postfix({"avg_loss": f"{avg_loss:.4f}", "current_loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


if __name__ == "__main__":
    train_pairs = read_split(TRAIN_DATA)
    valid_pairs = read_split(VALID_DATA)

    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    train_pairs = [(sp.encode(src), sp.encode(tgt, add_bos=True, add_eos=True)) for src, tgt in train_pairs]
    valid_pairs = [(sp.encode(src), sp.encode(tgt, add_bos=True, add_eos=True)) for src, tgt in valid_pairs]

    train_dataset = UQADataset(train_pairs)
    valid_dataset = UQADataset(valid_pairs)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: collate_fn(x, PAD_IDX)
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: collate_fn(x, PAD_IDX)
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

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = CrossEntropyLoss(ignore_index=PAD_IDX)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULE_PATIENCE,
        cooldown=SCHEDULE_COOLDOWN,
        min_lr=MIN_LR
    )

    best_valid_loss = float("inf")
    epoch_train_losses, epoch_valid_losses = [], []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    training_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        TF_RATIO = tf_ratio(epoch)
        train_loss = train_step(train_dataloader, model, optimizer, criterion, DEVICE, PAD_IDX, TF_RATIO, CLIP)
        valid_loss = evaluation_step(valid_dataloader, model, criterion, DEVICE, PAD_IDX)

        scheduler.step(valid_loss)

        epoch_train_losses.append(train_loss)
        epoch_valid_losses.append(valid_loss)

        print(f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), BEST_MODEL)
            print(f"Best model saved with loss: {best_valid_loss:.4f}")

    training_end = time.time()
    wall_clock_seconds = training_end - training_start
    wall_clock_str = str(timedelta(seconds=int(wall_clock_seconds)))

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS + 1), epoch_train_losses, label="Train Loss")
    plt.plot(range(1, NUM_EPOCHS + 1), epoch_valid_losses, label="Valid Loss")
    plt.title("Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(FIGS_DIR / "loss_plot.png")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "model_configuration.md", "w", encoding="utf-8") as f:
        f.write("| | |\n")
        f.write("|---|---|\n")
        f.write(f"| Encoder / decoder type (LSTM or GRU) | LSTM / LSTM |\n")
        f.write(f"| Layers / embedding / hidden size | {NUM_LAYERS} / {EMBEDDING_DIM} / {HIDDEN_DIM} |\n")
        f.write(f"| Vocabulary size | {VOCAB_SIZE} |\n")
        f.write(f"| Trainable parameters | {num_parameters:,} |\n")
        f.write(f"| Optimiser, learning rate, schedule | Adam, lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}, ReduceLROnPlateau(factor={SCHEDULER_FACTOR}, patience={SCHEDULE_PATIENCE}) |\n")
        f.write(f"| Batch size, epochs, wall-clock, GPU | {BATCH_SIZE}, {NUM_EPOCHS}, {wall_clock_str}, {gpu_name} |\n")
