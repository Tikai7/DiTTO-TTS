import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EncodecModel, GPT2LMHeadModel
from components.EnCodec import EnCodec

class NeuralAudioCodec(nn.Module):
    """
    Neural Audio Codec with encoder, decoder, and cosine similarity for alignment loss.
    """

    def __init__(self, lambda_factor=0.1):
        super().__init__()

        self.language_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.audio_encoder = EnCodec(self.language_model.config.n_embd)
        self.audio_decoder = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.lambda_factor = lambda_factor

        print(f"[INFO] GPT2 Embedding dim : {self.language_model.config.n_embd}")
        
        for param in self.audio_decoder.parameters():
            param.requires_grad = False

    def forward(self, audio_input, text_input):
        audio_latents = self.audio_encoder(audio_input)

        with torch.no_grad():
            reconstructed_audio = self.audio_decoder.decode(audio_latents)

        text_embeddings = self.language_model.transformer.wte(text_input)

        cosine_similarity = F.cosine_similarity(audio_latents, text_embeddings, dim=-1)
        alignment_loss = 1 - cosine_similarity.mean()

        return {
            "reconstructed_audio" : reconstructed_audio, 
            "audio_latents" : audio_latents, 
            "alignment_loss": alignment_loss
        }

    def compute_total_loss(self, reconstruction_loss, alignment_loss):
        return reconstruction_loss + self.lambda_factor * alignment_loss
