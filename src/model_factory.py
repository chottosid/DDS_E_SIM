"""
Model creation utilities
"""
import torch
import torch.nn as nn
from .model import TransformerVAE
from .autoregressive_model import DNAErrorModel
import config


def create_model(model_type=None):
    """Create model instance based on config"""
    if model_type is None:
        model_type = config.MODEL_TYPE
    
    if model_type == 'vae':
        return TransformerVAE(
            input_dim=config.VOCAB_SIZE,
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_layers=config.NUM_LAYERS,
            latent_dim=config.LATENT_DIM,
            seq_len=config.MAX_SEQ_LEN
        )
    elif model_type == 'autoregressive':
        return DNAErrorModel(
            vocab_size=config.VOCAB_SIZE,
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            num_encoder_layers=config.NUM_LAYERS,
            num_decoder_layers=config.NUM_LAYERS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_checkpoint_dir(model_type=None):
    """Get checkpoint directory for model type"""
    if model_type is None:
        model_type = config.MODEL_TYPE
    
    if model_type == 'vae':
        return config.VAE_CHECKPOINT_DIR
    elif model_type == 'autoregressive':
        return config.AUTOREGRESSIVE_CHECKPOINT_DIR
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def setup_model_for_training(model_type=None):
    """
    Set up model with DataParallel if multiple GPUs are available
    
    Returns:
        Tuple of (model, device)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(model_type)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model.to(device)
    return model, device
