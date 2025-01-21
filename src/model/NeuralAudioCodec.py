import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EncodecModel, GPT2LMHeadModel
from components.EnCodec import EnCodec
from components.VectorQuantizer import VectorQuantizer

class NAC(nn.Module):
    """
    Neural Audio Codec with encoder, decoder, and alignment loss using cosine similarity.
    """
    def __init__(self, lambda_factor=0.1):
        super().__init__()

        self.language_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.audio_encoder = EnCodec(self.language_model.config.n_embd)
        self.audio_decoder = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.vector_quantizer = VectorQuantizer(self.audio_encoder.model.config.codebook_size, self.language_model.config.n_embd)
        self.lambda_factor = lambda_factor

        for param in self.audio_decoder.parameters():
            param.requires_grad = False

        for param in self.language_model.parameters():
            param.requires_grad = False

    def forward(self, text_input, audio_input, padding_mask_audio):
        """
        Forward pass for the neural audio codec model.
        
        Args:
            text_input (dict): Text input dictionary with "input_ids" and "attention_mask".
            audio_input (Tensor): Audio input tensor (e.g., waveform or spectrogram).
    
        Returns:
            dict: A dictionary containing reconstructed audio, audio latents, 
                  alignment loss, reconstruction loss, and total loss.
        """

        audio_latents, audio_scales = self.audio_encoder(audio_input, padding_mask_audio)
        audio_latents_quantized = self.vector_quantizer(audio_latents)
        with torch.no_grad():
            reconstructed_audio = self.audio_decoder.decode(audio_latents_quantized.unsqueeze(0), audio_scales=audio_scales, padding_mask=padding_mask_audio)[0]

        reconstruction_loss = F.mse_loss(reconstructed_audio, audio_input)


        max_length = self.language_model.config.n_positions
        lm_audio_latents = audio_latents[:, :, :max_length].mean(dim=1)
        with torch.no_grad():
            print(lm_audio_latents.shape, text_input["input_ids"].shape, text_input["attention_mask"].shape)
            lm_outputs = self.language_model(
                inputs_embeds=lm_audio_latents,
                attention_mask=text_input["attention_mask"],
                labels=text_input["input_ids"]
            )
        
        lm_loss = lm_outputs.loss

        total_loss = reconstruction_loss + self.lambda_factor * (lm_loss)

        return {
            "reconstructed_audio": reconstructed_audio,
            "audio_latents": audio_latents,
            "lm_loss": lm_loss,
            "reconstruction_loss": reconstruction_loss,
            "total_loss": total_loss
        }
    