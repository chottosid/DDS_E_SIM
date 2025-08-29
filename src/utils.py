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

pad_idx = list(char_to_onehot.values()).index(char_to_onehot['P'])
inv_char_map = {i: c for i, c in enumerate(char_to_onehot.keys())}

# Index to character mapping for autoregressive model
index_to_char = {
    0: 'A',
    1: 'C',
    2: 'G',
    3: 'T',
    4: 'S',
    5: 'E',
    6: 'P'
}

def onehot_encode_sequence(seq, char_to_onehot=char_to_onehot):
    return [char_to_onehot[char] for char in seq]

def prepare_sequence(sequence: str, allowable_len: int) -> str:
    if len(sequence) > allowable_len - 2:
        sequence = sequence[:allowable_len - 2]
    seq = 'S' + sequence + 'E'
    seq += 'P' * (allowable_len - len(seq))
    return seq

def decode_one_hot_sequence(seq_tensor, index_to_char_map=index_to_char):
    """
    Decode a one-hot encoded sequence tensor back to string
    
    Args:
        seq_tensor: Tensor of shape [seq_len, vocab_size]
        index_to_char_map: Dictionary mapping indices to characters
    
    Returns:
        Decoded string sequence
    """
    indices = seq_tensor.argmax(dim=-1).cpu().numpy()
    return ''.join(index_to_char_map[idx] for idx in indices)
