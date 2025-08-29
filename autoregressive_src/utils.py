import torch
import numpy as np

char_to_onehot = {
    'A': [1, 0, 0, 0, 0, 0, 0],
    'C': [0, 1, 0, 0, 0, 0, 0],
    'G': [0, 0, 1, 0, 0, 0, 0],
    'T': [0, 0, 0, 1, 0, 0, 0],
    'S': [0, 0, 0, 0, 1, 0, 0],
    'E': [0, 0, 0, 0, 0, 1, 0],
    'P': [0, 0, 0, 0, 0, 0, 1],
}

index_to_char = {
    0: 'A',
    1: 'C',
    2: 'G',
    3: 'T',
    4: 'S',
    5: 'E',
    6: 'P'
}

def decode_one_hot_sequence(seq_tensor, index_to_char=index_to_char):
    indices = seq_tensor.argmax(dim=-1).cpu().numpy()
    return ''.join(index_to_char[idx] for idx in indices)
