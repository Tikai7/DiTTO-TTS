import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

class MLSDataset(Dataset):
    def __init__(self, data_dir, max_text_token_length, sampling_rate=24000):
        """
        MLS Dataset for loading audio files and their corresponding transcripts.

        Args:
            data_dir (str): Path to the directory containing extracted MLS data.
            max_text_token_length (int): Maximum token length for text sequences.
            sampling_rate (int): Audio sampling rate.
        """
        self.data_dir = data_dir
        self.audio_dir = os.path.join(data_dir, "audio")  # Directory containing audio files (.opus)
        self.transcripts_file = os.path.join(data_dir, "transcripts.txt").replace('\\','/')  # Transcription file
        self.sampling_rate = sampling_rate
        self.max_text_token_length = max_text_token_length

        # Load transcripts and audio paths
        self.data = self.__load_data()

        # Initialize processors
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Load a sample at a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Contains "audio", "text", and "label".
        """
        audio_path, transcript = self.data[idx]

        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
            waveform = resampler(waveform)

        # Process audio
        processed_audio = self.processor(
            raw_audio=waveform.squeeze().numpy(), sampling_rate=self.sampling_rate, return_tensors="pt"
        )["input_values"].squeeze(0)

        # Tokenize text
        tokenized_text = self.tokenizer(
            transcript,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_token_length,
            return_tensors="pt",
        )

        # Compute audio duration (in seconds)
        duration = waveform.size(-1) / self.sampling_rate
        return {
            "audio": processed_audio,
            "text": tokenized_text,
            "label": torch.tensor(duration, dtype=torch.long),
        }

    def __load_data(self):
        """
        Load paths of audio files and their corresponding transcripts.

        Returns:
            list: List of tuples (audio_path, transcript).
        """
        data = []
        if not os.path.exists(self.transcripts_file):
            raise FileNotFoundError(f"Transcription file not found: {self.transcripts_file}")

        with open(self.transcripts_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")  # Assuming transcripts are tab-separated
                if len(parts) != 2:
                    continue
                audio_path, transcript = parts
                tab_audio_path = audio_path.split("_")
                audio_full_path = os.path.join(self.audio_dir, tab_audio_path[0] , tab_audio_path[1] , audio_path+'.opus').replace('\\','/')
                if os.path.exists(audio_full_path):
                    data.append((audio_full_path, transcript))
        return data

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle padding for audio tensors.
        """
        audio = [item["audio"].squeeze(0) for item in batch]  # Ensure audio is 1D
        text = {key: torch.cat([item["text"][key] for item in batch], dim=0) for key in batch[0]["text"]}
        labels = torch.stack([item["label"] for item in batch])
        audio_padded = pad_sequence(audio, batch_first=True)

        return {"audio": audio_padded, "text": text, "label": labels}

