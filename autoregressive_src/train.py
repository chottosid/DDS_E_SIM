import torch
import torch.nn as nn
from tqdm import tqdm

def train_step(model, optimizer, src, tgt, criterion=nn.CrossEntropyLoss()):
    optimizer.zero_grad()
    
    output = model(src, tgt[:, :-1])
    
    loss = criterion(
        output.reshape(-1, output.size(-1)),
        tgt[:, 1:].argmax(dim=-1).reshape(-1)
    )
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()

def validate(model, val_loader, criterion=nn.CrossEntropyLoss()):
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

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss(ignore_index=6)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for src, tgt in train_progress:
            src, tgt = src.to(device), tgt.to(device)
            loss = train_step(model, optimizer, src, tgt, criterion)
            train_loss += loss
            train_progress.set_postfix({"Train Loss": f"{loss:.4f}"})
        
        model.eval()
        val_loss = 0
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for src, tgt in val_progress:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt[:, :-1])
                loss = criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt[:, 1:].argmax(dim=-1).reshape(-1)
                )
                val_loss += loss.item()
                val_progress.set_postfix({"Val Loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.module.state_dict(), '/kaggle/working/best_model.pt')
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
