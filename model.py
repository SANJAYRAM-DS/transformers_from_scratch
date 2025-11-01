import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.droupout = nn.Dropout(dropout)
        
        #creating a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        #Creating a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange)(0, d_model, 1).float() * (-math.log(10000.0 / d_model))
        #Applying the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        #for cosine
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) #(1, seq_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.droupout(x)
    
