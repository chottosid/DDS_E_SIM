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

def onehot_encode_sequence(seq, char_to_onehot=char_to_onehot):
    return [char_to_onehot[char] for char in seq]

def prepare_sequence(sequence: str, allowable_len: int) -> str:
    if len(sequence) > allowable_len - 2:
        sequence = sequence[:allowable_len - 2]
    seq = 'S' + sequence + 'E'
    seq += 'P' * (allowable_len - len(seq))
    return seq
