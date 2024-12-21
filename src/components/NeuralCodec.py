import torch
import torch.nn as nn
from transformers import EncodecModel, AutoProcessor

class NeuralCodec(nn.Module):
    """
    Neural Codec pour embedder le spectre audio
    Utilisation de EnCodec au lieu de MEL-VAE présenté dans le papier car ce dernier a été obtenu en collaborant avec d'autres auteurs.
    EnCodec est plus simple à obtenir et est fourni dans la librairie HuggingFace (transformers).
    """
    def __init__(self, hidden_size):
        super(NeuralCodec, self).__init__()  
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.embedding_head = nn.Embedding(self.model.config.codebook_size, hidden_size) 

    def forward(self, X, sampling_rate=24000):
        # inputs = self.processor(raw_audio=X, sampling_rate=sampling_rate, return_tensors="pt")   
        inputs = {
            "input_values" : X
        }
        with torch.no_grad(): 
            # Les poids sont inchangés pour le modèle
            encoded_outputs = self.model.encode(inputs["input_values"])
        latents = encoded_outputs["audio_codes"]  
        projected_outputs = self.embedding_head(latents)
        return projected_outputs