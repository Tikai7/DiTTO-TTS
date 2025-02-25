import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm 

class MLSDataset(Dataset):
    def __init__(self, data_dir, max_text_token_length, sampling_rate=24000, nb_samples=None, tokenizer_model="google/byt5-small"):
        """
        MLS Dataset for loading audio files and their corresponding transcripts.

        Args:
            data_dir (str): Path to the directory containing extracted MLS data.
            max_text_token_length (int): Maximum token length for text sequences.
            sampling_rate (int): Audio sampling rate.
        """
        self.data_dir = data_dir
        self.audio_dir = os.path.join(data_dir, "audio_clean")  # Directory containing clean audio files  (.opus)
        self.transcripts_file = os.path.join(data_dir, "transcripts.txt").replace('\\','/')  # Transcription file
        self.tokenized_file = os.path.join(data_dir, f"tokenized_transcripts_{tokenizer_model.replace('/','_').replace('-','_')}.json")
        self.sampling_rate = sampling_rate
        self.max_text_token_length = max_text_token_length

        # Initialize processors
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load and preprocess data
        if os.path.exists(self.tokenized_file):
            self.data = self.__load_tokenized_data()
        else:
            self.data = self.__preprocess_and_save_data()

        if nb_samples is not None :
            self.data = self.data[:nb_samples]

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve a single dataset item."""
        audio_path, tokenized_text, duration = self.data[idx]

        if os.getcwd().split("/")[1] == "tempory": # only works on sorbonne's PPTI just tokenize well ur file and remove it
            audio_path = audio_path.replace(
                "C:/Cours-Sorbonne/M2/UE_DEEP/AMAL/Projet/data/mls_french_opus/mls_french_opus/train/audio_clean", 
                "/tempory/M2-DAC/UE_DEEP/AMAL/DiTTO-TTS/data/mls_french_opus/train/audio_clean"
            )
    
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
            waveform = resampler(waveform)

        
        # Process audio
        output_processor = self.processor(
            raw_audio=waveform.squeeze().numpy(), sampling_rate=self.sampling_rate, return_tensors="pt"
        )

        processed_audio = output_processor["input_values"].squeeze(0)
        padding_mask_audio = output_processor["padding_mask"].squeeze(0)

        duration = waveform.size(-1) / self.sampling_rate
        if not (10 <= duration <= 20):
            print(duration)
            raise ValueError(f"Duration {duration} out of bounds for index {idx}")

        duration = int(duration - 10)  # Shift duration to [0, 10]
        assert 0 <= duration <= 10, f"Mapped target {duration} out of bounds"

        return {
            "audio": processed_audio,
            "text": tokenized_text,
            "padding_mask_audio": padding_mask_audio,
            "label": torch.tensor(duration, dtype=torch.long),
        }

    def __preprocess_and_save_data(self):
        """Preprocess data and save tokenized transcripts."""
        data = []
        if not os.path.exists(self.transcripts_file):
            raise FileNotFoundError(f"Transcription file not found: {self.transcripts_file}")

        print("Tokenizing transcripts and saving results...")
        with open(self.transcripts_file, "r", encoding="utf-8", errors="replace") as f:
            for idx, line in tqdm(enumerate(f)):
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
                        # Tokenize transcript
                        tokenized_text = self.tokenizer(
                            transcript,
                            padding="max_length",
                            truncation=True,
                            max_length=self.max_text_token_length,
                            return_tensors="pt",
                        )
                        # Compute audio duration (dummy placeholder for now)
                        duration = 0  # Actual duration will be computed in __getitem__
                        data.append((audio_full_path, tokenized_text, duration))
                except UnicodeDecodeError as e:
                    print(f"[ERROR] UnicodeDecodeError in line {idx}: {e}")
                    continue

        # Save pre-tokenized data without overwriting the original transcript file
        with open(self.tokenized_file, "w", encoding="utf-8") as f:
            json.dump([(d[0], {k: v.tolist() for k, v in d[1].items()}, d[2]) for d in data], f)

        return data

    def __load_tokenized_data(self):
        """Load pre-saved tokenized transcripts."""
        print("Loading tokenized transcripts...")
        with open(self.tokenized_file, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        # Convert tokenized text back to tensors
        return [(d[0], {k: torch.tensor(v) for k, v in d[1].items()}, d[2]) for d in data]

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle padding and audio codes for EnCodec.
        """
        audio = [item["audio"].squeeze(0) for item in batch]  # Ensure audio is 1D
        audio_padded = pad_sequence(audio, batch_first=True)  # Pad audio tensors

        padding_masks = [item["padding_mask_audio"].squeeze(0) for item in batch]
        padding_masks_padded = pad_sequence(padding_masks, batch_first=True, padding_value=True)  # Padding mask

        text = {key: torch.cat([item["text"][key] for item in batch], dim=0) for key in batch[0]["text"]}
        labels = torch.stack([item["label"] for item in batch])

        return {
            "audio": audio_padded.unsqueeze(1),  
            "padding_mask_audio": padding_masks_padded,  
            "text": text,  
            "label": labels,  
        }
