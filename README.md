# DDS-E-SIM: DNA Error Simulation

A flexible framework for simulating DNA sequencing errors using deep learning models.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Choose your model in `config.py`:
   ```python
   MODEL_TYPE = 'vae'  # or 'autoregressive'
   ```

3. Run training:
   ```bash
   python main.py
   ```

## Project Structure

```
src/
├── data.py                    # Data loading for both models
├── model.py                   # VAE implementation  
├── autoregressive_model.py    # Autoregressive transformer
├── model_factory.py           # Model creation utilities
├── train.py                   # VAE training
├── autoregressive_train.py    # Autoregressive training
├── inference.py               # VAE inference
├── autoregressive_inference.py # Autoregressive inference
├── evaluation.py              # Analysis and evaluation tools
└── utils.py                   # Shared utilities

config.py                      # All configuration settings
main.py                        # Main entry point
```

## Models

**VAE**: Uses encoder-decoder architecture with latent space for learning sequence representations.

**Autoregressive**: Standard transformer that generates sequences token by token.

Switch between models by changing `MODEL_TYPE` in config.py.

## Analysis

The framework includes comprehensive error analysis:
- Insertion/deletion/substitution rates
- Position-wise error patterns  
- K-mer frequency analysis
- Substitution matrices

Results are automatically generated during inference.
