import torch
import torch.nn as nn
import pandas as pd
import gc
from autoregressive_src.data import prepare_dataset_from_files, create_data_loaders
from autoregressive_src.model import DNAErrorModel
from autoregressive_src.train import train_model
from autoregressive_src.inference import run_inference
from autoregressive_src.analysis import analyze_errors, plot_error_analysis, plot_error_rates
import autoregressive_config as config

def main():
    print("Preparing dataset...")
    prepare_dataset_from_files(
        config.CENTERS_PATH,
        config.CLUSTERS_PATH,
        config.OUTPUT_CSV_PATH
    )
    
    data = pd.read_csv(config.OUTPUT_CSV_PATH)
    print(f"Dataset loaded with {len(data)} sequences")
    
    max_len_noisy = data['Noisy_Read'].apply(len).max()
    min_len_noisy = data['Noisy_Read'].apply(len).min()
    avg_len_noisy = data['Noisy_Read'].apply(len).mean()
    
    print(f"Max Length of Noisy_Read: {max_len_noisy}")
    print(f"Min Length of Noisy_Read: {min_len_noisy}")
    print(f"Average Length: {avg_len_noisy:.2f}")
    
    train_loader, val_loader = create_data_loaders(
        config.OUTPUT_CSV_PATH,
        batch_size=config.BATCH_SIZE
    )
    
    model = DNAErrorModel(
        vocab_size=7,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
        lr=config.LEARNING_RATE
    )
    
    model = DNAErrorModel(
        vocab_size=7,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model = nn.DataParallel(model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print("Running inference...")
    original_seqs, read_seqs, output_seqs = run_inference(model, val_loader, device)
    
    print("Analyzing model output vs original sequences...")
    results = analyze_errors(original_seqs, output_seqs)
    
    print("Overall Error Analysis:")
    print(f"Total bases analyzed: {results['total_bases']}")
    print(f"Total errors: {results['total_errors']}")
    print(f"Total error rate: {results['total_error_rate']:.3f}")
    
    print("\nError Distribution:")
    for error_type, percentage in results['error_percentages'].items():
        count = results['error_counts'][error_type]
        print(f"{error_type.capitalize()}: {count} ({percentage:.2f}%)")
    
    plot_error_analysis(results)
    
    print("\nAnalyzing read sequences vs original sequences...")
    read_results = analyze_errors(original_seqs, read_seqs)
    plot_error_analysis(read_results)
    
    print("\nComparing error rates...")
    plot_error_rates(original_seqs, read_seqs, output_seqs)
    
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
