import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .utils import char_to_onehot, pad_idx


def get_model(model):
    """Get actual model from DataParallel wrapper"""
    return model.module if hasattr(model, 'module') else model


def train_step(model, optimizer, src, tgt, criterion=nn.CrossEntropyLoss()):
    """Single training step"""
    optimizer.zero_grad()
    
    # Forward pass
    output = model(src, tgt[:, :-1])  # Remove last target token for teacher forcing
    
    # Calculate loss (ignore padding)
    loss = criterion(
        output.reshape(-1, output.size(-1)),
        tgt[:, 1:].argmax(dim=-1).reshape(-1)  # Get target indices, shifted by 1
    )
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()


def validate(model, val_loader, criterion=nn.CrossEntropyLoss()):
    """Validate model"""
    model.eval()
    total_loss = 0
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(device)  
            tgt = tgt.to(device)
            
            output = model(src, tgt[:, :-1])
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt[:, 1:].argmax(dim=-1).reshape(-1)
            )
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train_autoregressive(model, train_loader, val_loader, optimizer, device, epochs=50, lr=1e-4, 
                        checkpoint_dir='checkpoints', save_every=5):
    """Train autoregressive model"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss(ignore_index=6)  # Ignore padding token ('P' = index6)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Training loop with tqdm
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for src, tgt in train_progress:
            src, tgt = src.to(device), tgt.to(device)
            loss = train_step(model, optimizer, src, tgt, criterion)
            train_loss += loss
            train_progress.set_postfix({"Train Loss": f"{loss:.4f}"})
        
        # Validation loop with tqdm
        val_loss = validate(model, val_loader, criterion)
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Save checkpoint
        if epoch % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': get_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
            }, os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt'))
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(get_model(model).state_dict(), best_model_path)
            print(f"New best model saved: {best_model_path}")
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
