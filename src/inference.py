import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from .utils import char_to_onehot, prepare_sequence, onehot_encode_sequence
from .train import get_model

def run_inference(model, test_path, output_csv_path, device, max_samples=None, batch_size=32):
    model.eval()
    
    test_df = pd.read_csv(test_path)
    test_inputs = test_df['input'].astype(str).values
    test_targets = test_df['target'].astype(str).values
    
    if max_samples:
        test_inputs = test_inputs[:max_samples]
        test_targets = test_targets[:max_samples]
    
    results = []
    num_batches = (len(test_inputs) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Inference"):
        batch_inputs = test_inputs[i * batch_size:(i + 1) * batch_size]
        batch_targets = test_targets[i * batch_size:(i + 1) * batch_size]
        
        prepared = [prepare_sequence(s, 147) for s in batch_inputs]
        x = torch.tensor([onehot_encode_sequence(s, char_to_onehot) for s in prepared], dtype=torch.float)
        x = x.to(device)
        
        with torch.no_grad():
            memory, mu, logvar, z = model(x)
            B = x.size(0)
            prefix = torch.zeros(B, 1, 7, device=device)
            prefix[:,0,char_to_onehot['S'].index(1)] = 1
            
            for j in range(1, 147):
                logits = get_model(model).decode(prefix, memory, z)
                next_token = logits[:,-1,:].argmax(dim=-1)
                if (next_token == char_to_onehot['E'].index(1)).all():
                    break
                onehot = F.one_hot(next_token, 7).float().unsqueeze(1)
                prefix = torch.cat([prefix, onehot], dim=1)
            
            predictions = []
            for seq in prefix.cpu().argmax(dim=-1).tolist():
                chars = [list(char_to_onehot.keys())[idx] for idx in seq]
                result = ''.join(chars).split('E')[0].replace('S','')
                predictions.append(result)
        
        for input_seq, target_seq, pred_seq in zip(batch_inputs, batch_targets, predictions):
            results.append({
                'input': input_seq,
                'target': target_seq,
                'prediction': pred_seq
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(results)} results to {output_csv_path}")
    return results_df
