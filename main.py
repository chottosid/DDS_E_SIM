import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import gc
from src.data import DNASequenceDataset
from src.model import TransformerVAE
from src.train import train_free, get_model
from src.inference import run_inference
import config

def main():
    print("CUDA Available:", torch.cuda.is_available())
    
    device_count = torch.cuda.device_count()
    print("Device Count:", device_count)
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nDevice {i}: {props.name}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} ({torch.cuda.device_count()} GPUs available)")

    model = TransformerVAE(input_dim=7, d_model=256, nhead=8, num_layers=3, latent_dim=128)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    dataset = DNASequenceDataset(csv_path=config.TRAIN_PATH, allowable_len=147)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                       num_workers=4 if torch.cuda.is_available() else 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_free(model, loader, optimizer, device, epochs=config.NUM_EPOCHS, beta=config.BETA, 
               checkpoint_dir=config.CHECKPOINT_DIR, save_every=2)

    final_model_path = os.path.join(config.CHECKPOINT_DIR, 'final_model.pt')
    torch.save(get_model(model).state_dict(), final_model_path)

    results_df = run_inference(model, config.TEST_PATH, config.INFERENCE_OUTPUT_PATH, device, max_samples=100)
    print(results_df.head(5))
    
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
