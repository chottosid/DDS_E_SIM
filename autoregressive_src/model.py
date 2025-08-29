import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=120):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class DNAErrorModel(nn.Module):
    def __init__(
        self,
        vocab_size=7,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        self.input_embed = nn.Linear(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, vocab_size)
        )
        
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_mask=None,
        memory_key_padding_mask=None,
    ):
        if tgt_mask is None:
            tgt_mask = self.create_mask(tgt.size(1)).to(tgt.device)
        
        src = self.pos_encoder(self.input_embed(src))
        tgt = self.pos_encoder(self.input_embed(tgt))
        
        output = self.transformer(
            src, tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        return self.output_proj(output)
    
    def generate(self, src, max_len=None, temperature=1.0):
        if max_len is None:
            max_len = src.size(1)
        
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        with torch.no_grad():
            memory = self.transformer.encoder(
                self.pos_encoder(self.input_embed(src))
            )
            
            start_token = torch.zeros(batch_size, 1, src.size(-1)).to(device)
            start_token[:, :, 4] = 1
            
            cur_seq = start_token
            for i in range(max_len - 1):
                tgt_mask = self.create_mask(cur_seq.size(1)).to(device)
                out = self.transformer.decoder(
                    self.pos_encoder(self.input_embed(cur_seq)),
                    memory,
                    tgt_mask=tgt_mask
                )
                
                logits = self.output_proj(out[:, -1:]) / temperature
                probs = F.softmax(logits, dim=-1)
                
                next_token = torch.multinomial(probs.squeeze(1), 1)
                
                cur_seq = torch.cat([cur_seq, F.one_hot(next_token, num_classes=self.output_proj[-1].out_features).float()], dim=1)
                
                if (next_token == 5).any():
                    break
            
            return cur_seq
