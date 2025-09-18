# my_vae_library/__init__.py
from .vae import Sampling, VAE, build_encoder, build_decoder

# Optional: Define what should be imported with "from my_vae_library import *"
__all__ = [
    'Sampling',
    'VAE',
    'build_encoder',
    'build_decoder'
]