import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import get_config, get_weights_file_path
from dataset import BilingualDataset, causal_mask
from model import Seq2SeqTransformer, PositionalEncoding

# -------------------------
# Utilities: tokenizer build
# -------------------------
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not tokenizer_path.exists():
        # WordLevel tokenizer requires unk_token defined
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# -------------------------
# Data loading
# -------------------------
def get_ds(config):
    # load dataset (HuggingFace)
    pair = f"{config['lang_src']}-{config['lang_tgt']}"
    ds_raw = load_dataset('opus_books', pair, split='train')

    # Build / load tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # split
    train_size = int(0.9 * len(ds_raw))
    val_size = len(ds_raw) - train_size
    train_raw, val_raw = random_split(ds_raw, [train_size, val_size])

    train_ds = BilingualDataset(train_raw, tokenizer_src, tokenizer_tgt,
                                config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_raw, tokenizer_src, tokenizer_tgt,
                                config['lang_src'], config['lang_tgt'], config['seq_len'])

    # return dataloaders and tokenizers
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)
    return train_loader, val_loader, tokenizer_src, tokenizer_tgt

# -------------------------
# Greedy decode for validation
# -------------------------
def greedy_decode(model, src, src_mask, tokenizer_tgt, max_len, device):
    sos = tokenizer_tgt.token_to_id('[SOS]')
    eos = tokenizer_tgt.token_to_id('[EOS]')
    pad = tokenizer_tgt.token_to_id('[PAD]')

    model.eval()
    with torch.no_grad():
        memory = model.encode(src.to(device), src_mask.to(device))
        ys = torch.tensor([[sos]], dtype=torch.long, device=device)  # (1, 1)

        for i in range(max_len - 1):
            # create tgt mask for decoder (seq_len x seq_len) using torch.nn.Transformer expects float mask
            tgt_mask = torch.triu(torch.full((ys.size(1), ys.size(1)), float('-inf'), device=device), diagonal=1)
            out = model.decode(ys, memory, tgt_mask=tgt_mask)
            logits = model.project(out[:, -1:, :])  # (B, 1, V)
            next_token = logits.argmax(-1).item()
            ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)
            if next_token == eos:
                break

    return ys.squeeze(0).cpu().numpy().tolist()  # list of token ids

# -------------------------
# Validation printing
# -------------------------
def run_validation(model, val_loader, tokenizer_src, tokenizer_tgt, max_len, device, printer, global_step, writer, num_examples=2):
    model.eval()
    examples = 0
    with torch.no_grad():
        for batch in val_loader:
            src = batch['encoder_input'].to(device)
            src_mask = batch['encoder_mask'].to(device)
            # greedy decode
            out_ids = greedy_decode(model, src, src_mask, tokenizer_tgt, max_len, device)
            pred_text = tokenizer_tgt.decode(out_ids, skip_special_tokens=True)
            printer('-' * 80)
            printer(f"SOURCE: {batch['src_text'][0]}")
            printer(f"TARGET: {batch['tgt_text'][0]}")
            printer(f"PREDICTED: {pred_text}")
            examples += 1
            if examples >= num_examples:
                break

# -------------------------
# Training
# -------------------------
def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_ds(config)

    src_vocab = tokenizer_src.get_vocab_size()
    tgt_vocab = tokenizer_tgt.get_vocab_size()

    model = Seq2SeqTransformer(src_vocab, tgt_vocab,
                                d_model=config['d_model'],
                                nhead=8,
                                num_encoder_layers=6,
                                num_decoder_layers=6,
                                dim_feedforward=2048,
                                dropout=0.1,
                                max_len=config['seq_len']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    pad_idx = tokenizer_tgt.token_to_id('[PAD]')
    if pad_idx is None:
        raise ValueError("Target tokenizer must have [PAD] token")

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)

    writer = SummaryWriter(config['experiment_name'])

    global_step = 0
    start_epoch = 0
    if config.get('preload'):
        fname = get_weights_file_path(config, config['preload'])
        print("Loading weights from", fname)
        st = torch.load(fname, map_location=device)
        model.load_state_dict(st['model_state_dict'])
        optimizer.load_state_dict(st['optimizer_state_dict'])
        start_epoch = st['epoch'] + 1
        global_step = st.get('global_step', 0)

    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}")
        for batch in pbar:
            src = batch['encoder_input'].to(device)        # (B, S)
            tgt = batch['decoder_input'].to(device)       # (B, S)
            label = batch['label'].to(device)             # (B, S)

            # prepare masks for transformer:
            # src_key_padding_mask: True where padding -> transformer expects that
            src_key_padding_mask = ~batch['encoder_mask'].squeeze(0).squeeze(1).to(device)  # (B, S)
            # tgt_key_padding_mask:
            tgt_key_padding_mask = ~batch['decoder_mask'].squeeze(0).any(dim=1).to(device)  # (B, S) approximate
            # tgt_mask: (T, T) float mask where masked positions are -inf
            T = tgt.size(1)
            tgt_mask = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)

            optimizer.zero_grad()
            memory = model.encode(src, batch['encoder_mask'].to(device))
            out = model.decode(tgt, memory, tgt_mask=tgt_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=src_key_padding_mask)
            logits = model.project(out)  # (B, T, V)

            # compute loss: flatten
            loss = loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))
            loss.backward()
            optimizer.step()

            writer.add_scalar('train/loss', loss.item(), global_step)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1

        # validation at epoch end
        run_validation(model, val_loader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                        lambda msg: pbar.write(msg), global_step, writer, num_examples=3)

        # save checkpoint
        fname = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, fname)
        print("Saved:", fname)

    writer.close()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    cfg = get_config()
    train_model(cfg)
