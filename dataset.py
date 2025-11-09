import torch

def causal_mask(size: int, device=None):
    """Create a causal mask for decoder: (1, size, size) with 1s where allowed."""
    mask = torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)
    return ~mask  # True where allowed (including diagonal)

class BilingualDataset(torch.utils.data.Dataset):
    """
    Wrap a HuggingFace/datasets Dataset-like object `ds` (supports __len__ and __getitem__).
    Uses tokenizers.Tokenizer instances for src/tgt.
    """

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # token ids
        self.SOS = self.tokenizer_tgt.token_to_id('[SOS]')
        self.EOS = self.tokenizer_tgt.token_to_id('[EOS]')
        self.PAD = self.tokenizer_tgt.token_to_id('[PAD]')

        # if tokenizers did not return ids for special tokens, raise clearly
        if None in (self.SOS, self.EOS, self.PAD):
            raise ValueError("Tokenizer must contain special tokens: [SOS], [EOS], [PAD]")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        pair = self.ds[index]
        src_text = pair['translation'][self.src_lang]
        tgt_text = pair['translation'][self.tgt_lang]

        # encode -> python list of ints
        src_ids = self.tokenizer_src.encode(src_text).ids
        tgt_ids = self.tokenizer_tgt.encode(tgt_text).ids

        # Reserve space for special tokens
        # Source: [SOS] + src_ids + [EOS] + pads
        # Decoder input: [SOS] + tgt_ids + pads
        # Label: tgt_ids + [EOS] + pads
        if len(src_ids) + 2 > self.seq_len or len(tgt_ids) + 1 > self.seq_len:
            raise ValueError("Sentence too long for seq_len")

        enc = [self.SOS] + src_ids + [self.EOS] + [self.PAD] * (self.seq_len - (len(src_ids) + 2))
        dec_in = [self.SOS] + tgt_ids + [self.PAD] * (self.seq_len - (len(tgt_ids) + 1))
        label = tgt_ids + [self.EOS] + [self.PAD] * (self.seq_len - (len(tgt_ids) + 1))

        encoder_input = torch.tensor(enc, dtype=torch.long)
        decoder_input = torch.tensor(dec_in, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        # Masks:
        # encoder_mask: (1, 1, seq_len) True where not PAD
        encoder_mask = (encoder_input != self.PAD).unsqueeze(0).unsqueeze(0)  # (1,1,seq_len)
        # decoder_mask: causal combined with padding mask -> (1, seq_len, seq_len)
        padding_mask = (decoder_input != self.PAD).unsqueeze(0)  # (1, seq_len)
        caus = causal_mask(self.seq_len)
        # final decoder mask (seq_len, seq_len) boolean where allowed
        decoder_mask = caus & padding_mask.transpose(0, 1) & padding_mask  # uses broadcasting
        # For compatibility we will return decoder_mask shaped (1, seq_len, seq_len)
        decoder_mask = decoder_mask.unsqueeze(0)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask.to(torch.bool),
            "decoder_mask": decoder_mask.to(torch.bool),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
