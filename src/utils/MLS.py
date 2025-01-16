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
        try:
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

            # Handle encoding issues in transcript
            transcript = transcript.encode("utf-8", errors="replace").decode("utf-8", errors="replace")

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
        except UnicodeDecodeError as e:
            print(f"[ERROR] UnicodeDecodeError at index {idx}: {e}")
            print(f"Audio path: {audio_path}")
            print(f"Transcript: {transcript}")
            raise e
        
        except Exception as e:
            print(f"[ERROR] Unexpected error at index {idx}: {e}")
            print(f"Audio path: {audio_path}")
            return None  # Optionally skip problematic samples


    def __load_data(self):
        data = []
        if not os.path.exists(self.transcripts_file):
            raise FileNotFoundError(f"Transcription file not found: {self.transcripts_file}")

        with open(self.transcripts_file, "r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                try:
                    parts = line.strip().split("\t")
                    if len(parts) != 2:
                        continue
                    audio_path, transcript = parts
                    tab_audio_path = audio_path.split("_")
                    audio_full_path = os.path.join(
                        self.audio_dir, tab_audio_path[0], tab_audio_path[1], audio_path + ".opus"
                    ).replace("\\", "/")
                    if os.path.exists(audio_full_path):
                        data.append((audio_full_path, transcript))
                except UnicodeDecodeError as e:
                    print(f"[ERROR] UnicodeDecodeError in line {idx}: {e}")
                    continue
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

        return {"audio": audio_padded.unsqueeze(1), "text": text, "label": labels}

