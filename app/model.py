import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        """encoder initializer

        Args:
            input_size (int): size of the input
            hidden_size (int): size of the hidden state
            num_layers (int): number of layers in the lstm
            dropout (float): dropout rate
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
        )
        self.hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x, lengths):
        """forward pass of the encoder

        Args:
            x (tensor): input tensor of shape (s_seq_len, batch_size, embedding_dim)
            lengths (tensor): a tensor of shape (batch_size,) containing the actual lengths of the sequences in the batch

        Returns:
            tuple: a tuple containing the projected sequence hidden states of shape (s_seq_len, batch_size, hidden_size), final projected hidden and cell states of shape (num_layers, batch, hidden_size)
        """
        x = pack_padded_sequence(x, lengths.cpu(), enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.lstm(x)
        outputs, _ = pad_packed_sequence(packed_outputs)

        return self.hidden(outputs), self.hidden(torch.cat((hidden[::2], hidden[1::2]), dim=2)), self.cell(torch.cat((cell[::2], cell[1::2]), dim=2))


class LuongAttention(nn.Module):

    def __init__(self):
        """initializer of luong's global attention with dot score
        """
        super().__init__()

    def forward(self, enc_hiddens, dec_hidden, mask):
        """computing the context vector according to luong's global attention with dot score

        Args:
            enc_hiddens (tensor): encoder hidden states of shape (s_seq_len, batch_size, hidden_size)
            dec_hidden (tensor): decoder hidden state of shape (batch_size, hidden_size)
            mask (tensor): a tensor of shape (batch_size, s_seq_len) containing 1s for valid positions and 0s for padded positions
            
        Returns:
            tensor: context vector of shape (batch_size, hidden_size)
        """
        scores = enc_hiddens.transpose(0, 1) @ dec_hidden.unsqueeze(2)
        scores = scores.masked_fill(mask.unsqueeze(2) == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=1)
        context = attn_weights.transpose(1, 2) @ enc_hiddens.transpose(0, 1)

        return context.squeeze(1)
    

class Decoder(nn.Module):

    def __init__(self, attention, vocab_size, input_size, hidden_size, num_layers, dropout):
        """initializer of the decoder with attention

        Args:
            attention (LuongAttention): an instance of the LuongAttention class
            vocab_size (int): size of the vocabulary
            input_size (int): size of the input embeddings
            hidden_size (int): size of the hidden states
            num_layers (int): number of layers in the lstm
            dropout (float): dropout rate
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.attention = attention
        self.h_att_layer = nn.Linear(2 * hidden_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, y_prev, h_att_prev, h_prev, c_prev, encoder_hiddens, mask):
        """forward pass of the decoder with attention

        Args:
            y_prev (tensor): previous embedded output of the decoder of shape (batch_size, embedding_dim)
            h_att_prev (tensor): previous attention-hidden state of shape (batch_size, hidden_size)
            h_prev (tensor): previous hidden state of the lstm of shape (num_layers, batch_size, hidden_size)
            c_prev (tensor): previous cell state of the lstm of shape (num_layers, batch_size, hidden_size)
            encoder_hiddens (tensor): encoder hidden states of shape (s_seq_len, batch_size, hidden_size)
            mask (tensor): a tensor of shape (batch_size, s_seq_len) containing 1s for valid positions and 0s for padded positions

        Returns:
            tuple: a tuple containing the logit output of shape (batch_size, vocab_size), attention-hidden state of shape (batch_size, hidden_size), hidden state of shape (num_layers, batch_size, hidden_size), and cell state of shape (num_layers, batch_size, hidden_size)
        """
        x = torch.cat((y_prev, h_att_prev), dim=1).unsqueeze(0)
        output, (hidden, cell) = self.lstm(x, (h_prev, c_prev))

        context = self.attention(encoder_hiddens, output.squeeze(0), mask)
        h_att = F.tanh(self.h_att_layer(torch.cat((context, output.squeeze(0)), dim=1)))
        logits = self.out_layer(h_att)

        return logits, h_att, hidden, cell


class Seq2Seq(nn.Module):

    def __init__(self, device, pad_idx, sos_idx, eos_idx, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        """initializer of the seq2seq model with attention

        Args:
            device (torch.device): device to run the model on
            pad_idx (int): index of the padding token
            sos_idx (int): index of the start-of-sequence token
            eos_idx (int): index of the end-of-sequence token
            vocab_size (int): size of the vocabulary
            embedding_dim (int): dimension of the embedding layer
            hidden_dim (int): dimension of the hidden layer
            num_layers (int): number of layers in the lstm
            dropout (float): dropout probability
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.attention = LuongAttention()
        self.encoder = Encoder(embedding_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(self.attention, vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def forward(self, x, lengths, y, teacher_forcing_ratio=0.5):
        """complete end to end forward pass of seq2seq model

        Args:
            x (tensor): input sequence of shape (s_seq_len, batch_size)
            lengths (tensor): lengths of the input sequences
            y (tensor): target sequence of shape (t_seq_len, batch_size)
            teacher_forcing_ratio (float, optional): probability of using teacher forcing, defaults to 0.5

        Returns:
            tuple: a tuple containing the output logits of shape (t_seq_len - 1, batch_size, vocab_size)
        """
        embedded_x = self.dropout(self.embedding(x))
        encoder_hiddens, hidden, cell = self.encoder(embedded_x, lengths)

        outputs = torch.zeros((y.size(0) - 1, y.size(1), self.decoder.out_layer.out_features), device=self.device)

        mask = (x != self.pad_idx).transpose(0, 1)
        y_prev_token = y[0]
        h_att = torch.zeros((x.size(1), hidden.size(2)), device=self.device)

        for t in range(1, y.size(0)):
            embedded_y_prev = self.dropout(self.embedding(y_prev_token))
            logits, h_att, hidden, cell = self.decoder(embedded_y_prev, h_att, hidden, cell, encoder_hiddens, mask)
            outputs[t - 1] = logits
            y_prev_token = y[t] if torch.rand(1) < teacher_forcing_ratio else logits.argmax(1)

        return outputs

    @torch.no_grad()
    def decode(self, x, lengths, max_len, beam_size=1):
        """decodes the input sequence into a list of predicted sequences

        Args:
            x (tensor): input sequence of shape (s_seq_len, batch_size)
            lengths (tensor): lengths of the input sequences shape is (batch_size,)
            max_len (int): maximum length of the output sequences
            beam_size (int, optional): number of beams to use for decoding defaults is 1 (greedy)

        Returns:
            list: a list of predicted sequences, each sequence is a list of token indices of shape (batch_size, max_len)
        """
        self.eval()

        embedded_x = self.embedding(x)
        enc_hiddens, enc_hidden, enc_cell = self.encoder(embedded_x, lengths)

        mask = (x != self.pad_idx).transpose(0, 1)

        results = []

        for b in range(x.size(1)):
            enc_hiddens_b = enc_hiddens[:, b : b + 1, :]
            enc_hidden_b = enc_hidden[:, b : b + 1, :]
            enc_cell_b = enc_cell[:, b : b + 1, :]
            mask_b = mask[b : b + 1, :]
            h_att_b = torch.zeros((1, enc_hidden.size(2)), device=self.device)

            beams = [(
                0.0,
                [self.sos_idx],
                h_att_b,
                enc_hidden_b,
                enc_cell_b,
                False
            )]

            for _ in range(max_len):
                if all(beam[-1] for beam in beams):
                    break

                new_beams = []
                for score, tokens, h_att_prev, h_prev, c_prev, finished in beams:
                    if finished:
                        new_beams.append((score, tokens, h_att_prev, h_prev, c_prev, finished))
                        continue


                    y_prev_token = torch.tensor([tokens[-1]], device=self.device)
                    embedded_y_prev = self.embedding(y_prev_token)
                    logits, h_att, h, c = self.decoder(embedded_y_prev, h_att_prev, h_prev, c_prev, enc_hiddens_b, mask_b)
                    logp = F.log_softmax(logits, dim=1).squeeze(0)
                    topk_logp, topk_idxs = logp.topk(beam_size)

                    for logp, idx in zip(topk_logp.tolist(), topk_idxs.tolist()):
                        new_token = tokens + [idx]
                        new_score = score + logp
                        new_finished = finished or (idx == self.eos_idx)
                        new_beams.append((new_score, new_token, h_att, h, c, new_finished))

                beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]

            best_beam = max(beams, key=lambda x: x[0])
            results.append(best_beam[1][1:-1] if best_beam[-1] else best_beam[1][1:])

        return results
