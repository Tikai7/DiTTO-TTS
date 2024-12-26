import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer
from components.NeuralCodec import EnCodec
from utils.Config import Config

class MLS:
    def __init__(self, max_text_token_length, nb_samples=100, split="train", batch_size=32, sampling_rate=24000):
        mls = load_dataset(
            "facebook/multilingual_librispeech", 
            "french", 
            streaming=True
        )[split]  
        mls = mls.take(nb_samples)
        self.dataloader = DataLoader(mls, batch_size=batch_size, collate_fn=self.__collate_fn)
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
        self.sampling_rate = sampling_rate
        self.max_text_token_length = max_text_token_length
        self.audio_encoder = EnCodec(Config.EMBEDDING_DIM)

    def loader(self):
        return iter(self.dataloader)

    def __collate_fn(self, batch):
        audio = [
            self.processor(
                raw_audio=sample["audio"]["array"],
                sampling_rate=self.sampling_rate,
                return_tensors="pt"
            )["input_values"].squeeze(0)
            for sample in batch
        ]
        labels = [
            torch.ceil(torch.tensor(sample["audio_duration"])) for sample in batch
        ]
        text = [sample["transcript"] for sample in batch]
        model_inputs = self.tokenizer(
            text,  
            padding="longest",  
            truncation=True, 
            max_length=self.max_text_token_length, 
            return_tensors="pt" 
        )
        token_lengths = [a.size(-1) for a in audio]
        max_length = max(token_lengths)
        padded_audio = torch.stack([
            torch.nn.functional.pad(a, (0, max_length - a.size(-1))) for a in audio
        ])
        return {"audio": padded_audio, "text": model_inputs, "labels": torch.stack(labels)}
