import torch
import torch.nn as nn
from components.NeuralCodec import EnCodec
from components.TextEncoder import ByT5


class SLP(nn.Module):
    """
    Speech Length Predictor (SLP):
    - Encodes text and audio into embeddings.
    - Applies cross-attention between text embeddings and audio embeddings
      to predict the number of audio tokens required for generation.
    - Enables variable-length audio generation instead of using padding or silence.
    """
    def __init__(self, max_audio_token_length, nhead=4, num_layers=4):
        super().__init__()
        self.text_encoder = ByT5()
        self.hidden_size = self.text_encoder.model.config.d_model
        self.audio_encoder = EnCodec(self.hidden_size)
        
        # Transformer Decoder for cross-attention between text and audio embeddings
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hidden_size,       
                nhead=nhead,                   
                dim_feedforward=self.hidden_size * nhead,
                batch_first=True
            ),
            num_layers=num_layers
        )
        # Outputs a distribution over lengths up to `max_audio_token_length`
        self.length_predictor = nn.Linear(self.hidden_size, max_audio_token_length)

    def forward(self, text, audio):
        """
        Forward pass for the SLP model.

        Args:
            text (list of str): Input text sequences.
            audio (Tensor): Input raw audio sequences (batch of waveforms).

        Returns:
            Tensor: Predicted distribution over audio lengths for each input in the batch.
        """

        z_text = self.text_encoder(text)
        z_audio = self.audio_encoder(audio)
        z_audio = z_audio.view(z_audio.size(0), -1, z_audio.size(-1))  # Combine codebook et temporal length

        tgt_mask = self.generate_causal_mask(z_audio.size(1), z_audio.device)

        # Transformer Decoder to apply cross-attention between audio and text embeddings
        z_audio_decoded = self.transformer(z_audio, z_text, tgt_mask=tgt_mask)

        # Use the last token's embedding to predict the audio length
        lengths = self.length_predictor(z_audio_decoded[:, -1, :])

        return lengths

    @staticmethod
    def generate_causal_mask(size, device):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        mask = mask.to(device)
        return mask
