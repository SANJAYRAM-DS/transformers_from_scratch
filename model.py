import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :].to(x.dtype)
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """
    Small wrapper around torch.nn.Transformer that provides:
        - token embeddings (src & tgt)
        - positional encodings
        - encode/decode helpers
        - projection to logits
    """

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512, nhead: int = 8,
                    num_encoder_layers: int = 6, num_decoder_layers: int = 6, dim_feedforward: int = 2048,
                    dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model

        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                            num_encoder_layers=num_encoder_layers,
                                            num_decoder_layers=num_decoder_layers,
                                            dim_feedforward=dim_feedforward,
                                            dropout=dropout,
                                            batch_first=True)  # use batch_first for (B, S, E)

        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        # src: (B, S)
        src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        # transformer expects src_key_padding_mask shaped (B, S) with True for padded positions,
        # but our encoder_mask is True where valid; invert it:
        if src_mask is not None:
            src_key_padding_mask = ~src_mask.squeeze(0).squeeze(0)  # (B, S)
        else:
            src_key_padding_mask = None
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        # tgt_mask: (T, T) boolean where True means allowed (we will pass proper float mask below)
        # torch.nn.Transformer expects a float mask with float('-inf') where masked. We'll convert in training loop if needed.
        return self.transformer.decoder(tgt_emb, memory,
                                        tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # convenience method
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, tgt_mask=tgt_mask,
                            tgt_key_padding_mask=None,
                            memory_key_padding_mask=None)
        logits = self.generator(out)  # (B, T, V)
        return logits

    def project(self, x):
        return self.generator(x)
