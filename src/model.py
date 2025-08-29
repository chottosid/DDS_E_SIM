import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import char_to_onehot

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=147):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0,:,0::2] = torch.sin(pos * div)
        pe[0,:,1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerVAE(nn.Module):
    def __init__(self, input_dim=7, d_model=256, nhead=8, num_layers=3, latent_dim=128, seq_len=147):
        super().__init__()
        self.enc_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, seq_len)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_var = nn.Linear(d_model, latent_dim)
        
        self.dec_proj = nn.Linear(input_dim, d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.output_layer = nn.Linear(d_model, input_dim)

    def encode(self, x):
        h = self.enc_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return h, self.fc_mu(pooled), self.fc_var(pooled)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, prefix, memory, z):
        y = self.dec_proj(prefix)
        y = self.pos_enc(y)
        z_proj = self.latent_proj(z).unsqueeze(1).expand(-1, y.size(1), -1)
        tgt = y + z_proj
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        dec = self.decoder(tgt, memory, tgt_mask=mask)
        return self.output_layer(dec)

    def forward(self, x):
        memory, mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return memory, mu, logvar, z
