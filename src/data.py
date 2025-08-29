import pandas as pd
import torch
from torch.utils.data import Dataset
from .utils import onehot_encode_sequence, prepare_sequence, char_to_onehot, pad_idx

class DNASequenceDataset(Dataset):
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
