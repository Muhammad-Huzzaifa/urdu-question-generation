from config import *
import sentencepiece as spm


class UrduTokenizer:
    def __init__(self, model_path=TOKENIZER_MODEL):
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))

    def encode(self, text):
        return self.processor.encode(text)

    def decode(self, token_ids):
        return self.processor.decode(token_ids)