import torch
import torch.nn as nn
from NeuralCodec import EnCodec

class SLP(nn.Module):
    def __init__(self, hidden_size, max_length, nhead=4, num_layers=4):
        super(SLP, self).__init__()
        self.text_encoder =  ...
        self.audio_encoder =  EnCodec(hidden_size)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size*nhead
            ),
            num_layers=num_layers
        )
        self.length_predictor = nn.Linear(hidden_size*nhead, max_length)
    
    def forward(self, text, audio):
        z_text = self.text_encoder(text)
        z_audio = self.audio_encoder(audio)
        tgt_mask = self.generate_causal_mask(z_audio.size(0))
        z_audio_decoded = self.transformer(z_audio, z_text, tgt_mask=tgt_mask)
        lengths = self.length_predictor(z_audio_decoded[:, -1, :])
        return lengths

    @staticmethod
    def generate_causal_mask(size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask


