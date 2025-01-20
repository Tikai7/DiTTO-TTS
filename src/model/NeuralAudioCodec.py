import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EncodecModel, GPT2LMHeadModel
from components.EnCodec import EnCodec
from components.ByT5 import ByT5

class NeuralAudioCodec(nn.Module):
    """
    Neural Audio Codec with EnCodec (audio) and ByT5 (text) encoders.
    """
    def __init__(self, hidden_size, lambda_factor=0.1):
        super().__init__()
        self.audio_encoder = EnCodec(hidden_size)
        self.audio_decoder = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.text_encoder = ByT5()
        self.lambda_factor = lambda_factor

        # Freeze decoder parameters
        for param in self.audio_decoder.parameters():
            param.requires_grad = False

    def forward(self, audio_input, text_input):

        audio_latents = self.audio_encoder(audio_input)
        text_embeddings = self.text_encoder(text_input)

        # Reconstruct audio from latents
        with torch.no_grad():  # Decoder is frozen
            reconstructed_audio = self.audio_decoder.decode(audio_latents)

        # Compute cosine similarity for alignment loss
        cosine_similarity = F.cosine_similarity(audio_latents, text_embeddings.mean(dim=1), dim=-1)
        alignment_loss = 1 - cosine_similarity.mean()

        return reconstructed_audio, audio_latents, alignment_loss

    def compute_total_loss(self, reconstruction_loss, alignment_loss):
        """
        Combines the reconstruction and alignment losses.
        """
        return reconstruction_loss + self.lambda_factor * alignment_loss