from config import *
import torch
from app.model import Seq2Seq
from app.tokenizer import UrduTokenizer


class UrduQuestionGenerator:

    def __init__(self):
        """initialize the UrduQuestionGenerator with the model and tokenizer
        """
        self.device = torch.device(DEVICE)
        self.tokenizer = UrduTokenizer()
        self.model = Seq2Seq(
            device=self.device,
            pad_idx=PAD_IDX,
            sos_idx=SOS_IDX,
            eos_idx=EOS_IDX,
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(self.device)
        self.model.load_state_dict(torch.load(BEST_MODEL, map_location=self.device))
        self.model.eval()

    def generate(self, source):
        token_ids = self.tokenizer.encode(source)
        if not token_ids:
            raise ValueError("The source sentence could not be tokenized.")
        if len(token_ids) > MAX_SOURCE_LENGTH:
            raise ValueError(f"Use at most {MAX_SOURCE_LENGTH} source tokens.")

        source_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(1)
        lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=self.device)

        greedy_tokens, _ = self.model.decode(
            source_tensor, lengths, max_len=MAX_TARGET_LENGTH, beam_size=1
        )
        beam_tokens, _ = self.model.decode(
            source_tensor, lengths, max_len=MAX_TARGET_LENGTH, beam_size=BEAM_K
        )

        return {
            "source": source,
            "greedy": self.tokenizer.decode(greedy_tokens[0]).strip(),
            "beam": self.tokenizer.decode(beam_tokens[0]).strip(),
        }


generator = UrduQuestionGenerator()
