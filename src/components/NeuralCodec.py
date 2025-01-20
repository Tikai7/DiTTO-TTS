import torch
import torch.nn as nn
from transformers import EncodecModel, AutoProcessor

class EnCodec(nn.Module):
    """
    Neural Codec for embedding audio spectrograms : 
    - Uses EnCodec instead of MEL-VAE as described in the paper because MEL-VAE 
    - was developed in collaboration with other authors and may not be publicly accessible.
    - EnCodec, provided by HuggingFace's Transformers library, is a simpler and accessible alternative.
    """
    
    def __init__(self, hidden_size):
        super().__init__()
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.embedding_head = nn.Embedding(self.model.config.codebook_size, hidden_size) 
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
            
    def forward(self, X):
        """
        Forward pass to process raw audio input and produce latent embeddings.

        Args:
            X (Tensor): Raw audio input (batch of waveforms).
            sampling_rate (int): Sampling rate of the audio (default: 24000 Hz).

        Returns:
            Tensor: Latent audio embeddings projected to the target dimension.
        """
        # Gradients are disabled because the model weights are frozen
        with torch.no_grad(): 
            encoded_outputs = self.model.encode(X)
        
        latents = encoded_outputs["audio_codes"].squeeze(0)
        projected_outputs = self.embedding_head(latents)
        return projected_outputs
