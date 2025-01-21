import torch
import torch.nn as nn
from transformers import EncodecModel

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
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, X, padding_mask_audio=None):
        """
        Forward pass to process raw audio input and produce latent embeddings.

        Args:
            X (Tensor): Raw audio input (batch of waveforms).
        Returns:
            projected_outputs: Latent audio embeddings projected to the target dimension.
            audio_scales:  Scaling factor for each audio_codes input.

        """
        # Gradients are disabled because the model weights are frozen
        with torch.no_grad(): 
            encoded_outputs = self.model.encode(X, padding_mask_audio)
        
        latents = encoded_outputs["audio_codes"].squeeze(0)
        audio_scales = encoded_outputs["audio_scales"]
        projected_outputs = self.embedding_head(latents)

        return projected_outputs, audio_scales
