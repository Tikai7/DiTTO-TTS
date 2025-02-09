import torch
import torch.nn as nn 
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

class Speech2Text(nn.Module):
    def __init__(self, sampling_rate):
        super().__init__()
        self.model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")
        self.processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")
        self.sampling_rate = sampling_rate 
        self.token_fr = self.processor.tokenizer.lang_code_to_id["fr"]

    def forward(self, audio):
        audio = audio.cpu().numpy().squeeze()
        inputs = self.processor(audio, sampling_rate=self.sampling_rate, return_tensors="pt")
        generated_ids = self.model.generate(
            inputs["input_features"], attention_mask=inputs["attention_mask"],
            forced_bos_token_id=self.token_fr,  # Utilisation du token pour le français
        )
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return transcription
