import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .utils import char_to_onehot, pad_idx

def get_model(model):
    return model.module if hasattr(model, 'module') else model

def train_free(model, dataloader, optimizer, device, epochs=10, beta=0.1, checkpoint_dir='checkpoints', save_every=5):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float('inf')
    
    for ep in tqdm(range(1, epochs+1), desc="Training Progress"):
        model.train()
        total_loss = 0.0
        
        batch_progress = tqdm(dataloader, desc=f"Epoch {ep}/{epochs}", leave=False)
        for x_noisy, y in batch_progress:
            x_noisy, y = x_noisy.to(device), y.to(device)
            
            memory, mu, logvar, z = model(x_noisy)
            B, L, C = y.size()
            prefix = torch.zeros(B, 1, C, device=device)
            prefix[:,0,char_to_onehot['S'].index(1)] = 1
            
            batch_loss = 0.0
            
            for t in range(1, L):
                optimizer.zero_grad()
                
                logits = get_model(model).decode(prefix[:, :t], memory.detach(), z.detach())
                step_logits = logits[:, -1, :]
                target_idx = y.argmax(dim=-1)[:, t]
                step_loss = F.cross_entropy(step_logits, target_idx, ignore_index=pad_idx)
                
                if t == L-1:
                    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
                    step_loss = step_loss + beta * kl
                
                step_loss.backward()
                optimizer.step()
                
                batch_loss += step_loss.item()
                
                with torch.no_grad():
                    pred_idx = step_logits.argmax(dim=-1)
                    onehot = F.one_hot(pred_idx, C).float().unsqueeze(1)
                    if t < L-1:
                        new_prefix = torch.cat([prefix, onehot], dim=1)
                        prefix = new_prefix
            
            total_loss += batch_loss
            batch_progress.set_postfix(loss=batch_loss)
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {ep}/{epochs} — Loss: {avg_loss:.4f}")
        
        if ep % save_every == 0:
            torch.save({
                'epoch': ep,
                'model_state_dict': get_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(checkpoint_dir, f'model_epoch_{ep}.pt'))
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(get_model(model).state_dict(), best_model_path)
            print(f"New best model saved: {best_model_path}")
