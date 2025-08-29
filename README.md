DNA Sequence Prediction with Transformer VAE

This project implements a Transformer-based Variational Autoencoder for DNA sequence prediction tasks.

Setup

1. Install dependencies:
   pip install -r requirements.txt

2. Update data paths in config.py if needed

3. Run training and inference:
   python main.py

Project Structure

src/
├── __init__.py
├── data.py        - Dataset class and data loading
├── model.py       - Transformer VAE architecture
├── train.py       - Training loop with checkpointing
├── inference.py   - Model inference and result export
└── utils.py       - Utility functions and constants

config.py          - Hyperparameters and configuration
main.py           - Main entry point
requirements.txt  - Dependencies

Model Details

The model uses a Transformer encoder-decoder architecture with VAE latent space for DNA sequence generation. It processes sequences with one-hot encoding for nucleotides A, C, G, T plus special tokens for start, end, and padding.

Key features:
- Free-running autoregressive training
- KL divergence regularization 
- Positional encoding for sequence awareness
- Checkpointing and best model saving
- Batch inference with CSV export
