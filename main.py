import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import gc
import config
from src.data import DNASequenceDataset, create_datasets
from src.model_factory import setup_model_for_training, get_model_checkpoint_dir
from src.train import train_free, get_model
from src.autoregressive_train import train_autoregressive
from src.inference import run_inference
from src.autoregressive_inference import run_autoregressive_inference


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
    print(f"Model type: {config.MODEL_TYPE}")

    # Create model
    model, device = setup_model_for_training(config.MODEL_TYPE)
    checkpoint_dir = get_model_checkpoint_dir(config.MODEL_TYPE)

    if config.MODEL_TYPE == 'vae':
        # VAE training
        dataset = DNASequenceDataset(csv_path=config.TRAIN_PATH, allowable_len=config.MAX_SEQ_LEN)
        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                           num_workers=4 if torch.cuda.is_available() else 0)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        train_free(model, loader, optimizer, device, epochs=config.NUM_EPOCHS, beta=config.BETA, 
                   checkpoint_dir=checkpoint_dir, save_every=2)

        final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
        torch.save(get_model(model).state_dict(), final_model_path)

        # Run inference
        results_df = run_inference(model, config.TEST_PATH, config.INFERENCE_OUTPUT_PATH, device, max_samples=100)
        print("VAE inference completed")
        print(results_df.head(5))

    elif config.MODEL_TYPE == 'autoregressive':
        # Autoregressive training
        train_dataset, val_dataset = create_datasets(config.TRAIN_PATH, model_type='autoregressive', 
                                                   allowed_seq_len=config.MAX_SEQ_LEN)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=True
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)

        train_autoregressive(model, train_loader, val_loader, optimizer, device, 
                           epochs=config.NUM_EPOCHS, lr=config.LEARNING_RATE,
                           checkpoint_dir=checkpoint_dir, save_every=2)

        final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
        torch.save(get_model(model).state_dict(), final_model_path)

        # Run inference
        results_df = run_autoregressive_inference(model, config.TEST_PATH, config.INFERENCE_OUTPUT_PATH, 
                                                device, max_samples=100, temperature=config.TEMPERATURE)
        print("Autoregressive inference completed")
        print(results_df.head(5))

    else:
        raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")
    
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Training completed for {config.MODEL_TYPE} model")


if __name__ == "__main__":
    main()
