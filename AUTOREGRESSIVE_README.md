DNA Error Simulation with Transformer Model

This project implements a Transformer-based model for DNA sequencing error simulation and correction.

Setup

1. Install dependencies:
   pip install -r autoregressive_requirements.txt

2. Update data paths in autoregressive_config.py if needed

3. Run training and analysis:
   python autoregressive_main.py

Project Structure

autoregressive_src/
├── __init__.py
├── data.py        - Dataset preparation and loading
├── model.py       - DNA Error Transformer model
├── train.py       - Training loop and validation
├── inference.py   - Model inference
└── analysis.py    - Error analysis and visualization

autoregressive_config.py     - Configuration parameters
autoregressive_main.py      - Main execution script
autoregressive_requirements.txt - Dependencies

Model Details

The model uses a Transformer encoder-decoder architecture to simulate DNA sequencing errors. It processes DNA sequences with one-hot encoding and generates error-prone reads that mimic real nanopore sequencing data.

Features:
- Position-wise error analysis
- K-mer frequency comparison
- Substitution matrix analysis
- Comprehensive visualization
- Error rate comparison plots
