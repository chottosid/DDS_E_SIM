import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from .utils import onehot_encode_sequence, prepare_sequence, char_to_onehot, pad_idx


class DNASequenceDataset(Dataset):
    """Dataset for VAE approach - uses input/target columns with noise"""
    def __init__(self, csv_path, input_col='input', target_col='target', allowable_len=147, noise_factor=0.03):
        df = pd.read_csv(csv_path)
        self.inputs = df[input_col].astype(str).values
        self.targets = df[target_col].astype(str).values
        self.allowable_len = allowable_len
        self.noise_factor = noise_factor

    def __len__(self):
        return len(self.inputs)

    def add_noise(self, tensor):
        noise = self.noise_factor * torch.randn_like(tensor)
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __getitem__(self, idx):
        inp = prepare_sequence(self.inputs[idx], self.allowable_len)
        tgt = prepare_sequence(self.targets[idx], self.allowable_len)
        x = torch.tensor(onehot_encode_sequence(inp), dtype=torch.float)
        y = torch.tensor(onehot_encode_sequence(tgt), dtype=torch.float)
        return self.add_noise(x), y


class DNAAutoregressiveDataset(Dataset):
    """Dataset for autoregressive approach - uses center sequences and noisy reads"""
    def __init__(self, data_frame, indices, char_to_onehot, allowed_seq_len=108):
        self.data = data_frame.iloc[indices]
        self.char_to_onehot = char_to_onehot
        self.allowed_seq_len = allowed_seq_len
        
    def __len__(self):
        return len(self.data)
    
    def prepare_sequence(self, sequence):
        if len(sequence) > self.allowed_seq_len - 2:
            sequence = sequence[:self.allowed_seq_len - 2]
        sequence = 'S' + sequence + 'E'
        sequence += 'P' * (self.allowed_seq_len - len(sequence))
        return sequence
    
    def encode_sequence(self, seq):
        return [self.char_to_onehot[char] for char in seq]
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Encode input sequence (Center_Sequence)
        input_seq = self.prepare_sequence(row['Center_Sequence'])
        input_encoded = self.encode_sequence(input_seq)
        
        # Encode target sequence (Noisy_Read)
        target_seq = self.prepare_sequence(row['Noisy_Read'])
        target_encoded = self.encode_sequence(target_seq)
        
        return (torch.tensor(input_encoded, dtype=torch.float32),
                torch.tensor(target_encoded, dtype=torch.float32))


def create_datasets(csv_path, model_type='vae', **kwargs):
    """Create datasets based on model type"""
    if model_type == 'vae':
        return DNASequenceDataset(csv_path, **kwargs)
    elif model_type == 'autoregressive':
        df = pd.read_csv(csv_path)
        indices = np.arange(len(df))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        # Split into train/val
        split_idx = int(0.9 * len(indices))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_dataset = DNAAutoregressiveDataset(df, train_indices, char_to_onehot, 
                                               kwargs.get('allowed_seq_len', 108))
        val_dataset = DNAAutoregressiveDataset(df, val_indices, char_to_onehot,
                                             kwargs.get('allowed_seq_len', 108))
        
        return train_dataset, val_dataset
    else:
        raise ValueError(f"Unknown model type: {model_type}")
