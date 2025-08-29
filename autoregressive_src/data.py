import csv
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import char_to_onehot

def prepare_dataset_from_files(centers_path, clusters_path, output_csv_path):
    with open(centers_path, 'r') as centers_file:
        centers = [line.strip() for line in centers_file.readlines()]

    clusters = []
    current_cluster = []

    with open(clusters_path, 'r') as clusters_file:
        for line in clusters_file:
            line = line.strip()
            if all(char == '=' for char in line):
                if current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []
            else:
                if line:
                    current_cluster.append(line)
        if current_cluster:
            clusters.append(current_cluster)

    csv_data = []
    for center_id, (center_sequence, cluster) in enumerate(zip(centers, clusters)):
        for read in cluster:
            csv_data.append([center_id, center_sequence, read])

    with open(output_csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Center_ID', 'Center_Sequence', 'Noisy_Read'])
        writer.writerows(csv_data)

    print(f"Dataset prepared and saved as '{output_csv_path}'.")
    return output_csv_path

class DNADataset(Dataset):
    def __init__(self, data_frame, indices, char_to_onehot=char_to_onehot, allowed_seq_len=108):
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
        
        input_seq = self.prepare_sequence(row['Center_Sequence'])
        input_encoded = self.encode_sequence(input_seq)
        
        target_seq = self.prepare_sequence(row['Noisy_Read'])
        target_encoded = self.encode_sequence(target_seq)
        
        return (torch.tensor(input_encoded, dtype=torch.float32),
                torch.tensor(target_encoded, dtype=torch.float32))

def create_data_loaders(csv_path, batch_size=512, train_split=0.9):
    data = pd.read_csv(csv_path)
    indices = np.arange(len(data))
    np.random.seed(42)
    np.random.shuffle(indices)

    split_idx = int(train_split * len(indices))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = DNADataset(data, train_indices, char_to_onehot)
    val_dataset = DNADataset(data, val_indices, char_to_onehot)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader
