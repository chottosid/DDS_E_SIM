import torch
import pandas as pd
from torch.utils.data import DataLoader
from .utils import decode_one_hot_sequence, index_to_char
from .data import DNAAutoregressiveDataset


def run_autoregressive_inference(model, csv_path, output_path, device, max_samples=100, temperature=1.0):
    """Run inference using autoregressive model"""
    model.eval()
    
    # Load data
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)
    
    # Create dataset
    indices = list(range(len(df)))
    from .utils import char_to_onehot
    dataset = DNAAutoregressiveDataset(df, indices, char_to_onehot, allowed_seq_len=147)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    results = []
    
    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(loader):
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Generate sequences
            generated = model.generate(src, max_len=147, temperature=temperature)
            
            # Process each sequence in the batch
            for i in range(src.size(0)):
                # Decode sequences
                original_idx = batch_idx * loader.batch_size + i
                if original_idx >= len(df):
                    break
                    
                original_center = df.iloc[original_idx]['Center_Sequence']
                original_noisy = df.iloc[original_idx]['Noisy_Read']
                
                # Decode input (center sequence)
                input_seq = decode_one_hot_sequence(src[i], index_to_char)
                input_seq = input_seq.replace('S', '').replace('E', '').replace('P', '')
                
                # Decode target (noisy read)
                target_seq = decode_one_hot_sequence(tgt[i], index_to_char)
                target_seq = target_seq.replace('S', '').replace('E', '').replace('P', '')
                
                # Decode generated sequence
                generated_seq = decode_one_hot_sequence(generated[i], index_to_char)
                generated_seq = generated_seq.replace('S', '').replace('E', '').replace('P', '')
                
                results.append({
                    'original_center': original_center,
                    'original_noisy': original_noisy,
                    'input_processed': input_seq,
                    'target_processed': target_seq,
                    'generated_sequence': generated_seq
                })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Autoregressive inference results saved to {output_path}")
    
    return results_df
