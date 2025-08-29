import torch
import torch.nn as nn
from .utils import decode_one_hot_sequence, index_to_char

def run_inference(model, val_loader, device):
    original_seqs = []
    read_seqs = []
    output_seqs = []
    
    model.eval()
    
    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            generated = model.module.generate(src, max_len=147, temperature=1.0)
            
            for i in range(src.size(0)):
                read_seq_decoded = decode_one_hot_sequence(src[i], index_to_char)
                original_seq_decoded = decode_one_hot_sequence(tgt[i], index_to_char)
                output_seq_decoded = decode_one_hot_sequence(generated[i], index_to_char)
                
                read_seq_decoded = read_seq_decoded.replace('S', '').replace('E', '').replace('P', '')
                original_seq_decoded = original_seq_decoded.replace('S', '').replace('E', '').replace('P', '')
                output_seq_decoded = output_seq_decoded.replace('S', '').replace('E', '').replace('P', '')
                read_seqs.append(read_seq_decoded)
                original_seqs.append(original_seq_decoded)
                output_seqs.append(output_seq_decoded)

    print("Number of validation sequences:", len(original_seqs))
    print("\nSample outputs:")
    for i in range(3):
        print(f"Read:      {read_seqs[i]}")
        print(f"Original:  {original_seqs[i]}")
        print(f"Predicted: {output_seqs[i]}")
        print("-"*50)
    
    return original_seqs, read_seqs, output_seqs
