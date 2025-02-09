import os
import torch
import torchaudio
import torchaudio.transforms as T
from huggingface_hub import InferenceApi
from transformers import AutoProcessor, AutoTokenizer

from model.NeuralAudioCodec import NAC
from model.DiTTO import DiTTO
from utils.Config import ConfigDiTTO

class SpeechGenerator:
    def __init__(
        self,
        lambda_factor,
        nac_model_path="/tempory/M2-DAC/UE_DEEP/AMAL/DiTTO-TTS/src/params/NAC_epoch_20.pth",
        ditto_model_path="/tempory/M2-DAC/UE_DEEP/AMAL/DiTTO-TTS/src/params/DiTTO_epoch_20.pth",
        vocoder_repo_id="espnet/kan-bayashi_ljspeech_hifigan",  # Vocoder model on HuggingFace
        sample_rate=24000,
    ):
        """
        Initializes the SpeechGenerator with:
         - NAC model (for encoding/decoding audio)
         - Vocoder API (to convert mel-spectrogram to final waveform)
         - An audio processor to pre-process raw audio files (using facebook/encodec_24khz)
         - DiTTO model for conditioned generation.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load and initialize the DiTTO model.
        self.ditto_model = DiTTO(
            hidden_dim=ConfigDiTTO.HIDDEN_DIM,
            num_layers=ConfigDiTTO.NUM_LAYERS,
            num_heads=ConfigDiTTO.NUM_HEADS,
            time_dim=ConfigDiTTO.TIME_DIM,
            text_dim=ConfigDiTTO.TEXT_EMBED_DIM,
            diffusion_steps=ConfigDiTTO.DIFFUSION_STEPS,
            nac_model_path=nac_model_path,
        ).to(self.device)
        ditto_info = torch.load(ditto_model_path, map_location="cpu")
        self.ditto_model.load_state_dict(ditto_info["model_state_dict"])
        self.ditto_model.eval()

        # Load the NAC model.
        self.nac = NAC(lambda_factor=lambda_factor)
        nac_info = torch.load(nac_model_path, map_location="cpu")
        self.nac.load_state_dict(nac_info["model_state_dict"])
        self.nac.eval()
        self.nac.to(self.device)

        # Initialize the vocoder inference API.
        self.vocoder_api = InferenceApi(repo_id=vocoder_repo_id)

        # Mel-spectrogram transformation (parameters must match those expected by the vocoder).
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
        )

        # Audio processor for raw audio files.
        self.audio_processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.text_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    @torch.no_grad()
    def generate_speech_from_latents(self, audio_latents, audio_scales, padding_mask_audio):
        """
        Converts latents to an audio waveform using NAC decoder,
        then generates the final audio via the vocoder API.
        
        Args:
            audio_latents (Tensor): Latents from your model.
            audio_scales (Tensor): Scale factors.
            padding_mask_audio (Tensor): Mask for variable-length sequences.
        
        Returns:
            Tensor: Generated audio waveform.
        """
        # Decode latents into a raw waveform using NAC's audio_decoder.
        waveform = self.nac.audio_decoder.decode(
            audio_latents.unsqueeze(0),
            audio_scales=audio_scales,
            padding_mask=padding_mask_audio,
        )[0]

        # Compute the mel-spectrogram and apply logarithmic compression.
        mel_spectrogram = self.mel_transform(waveform)
        mel_spectrogram = torch.log1p(mel_spectrogram)

        # Convert to list format (expected by the HF inference API) and get final waveform.
        mel_np = mel_spectrogram.cpu().numpy().tolist()
        result = self.vocoder_api(inputs=mel_np)
        if "audio" not in result:
            print(result)
            raise ValueError(
                "The vocoder response does not contain the key 'audio'. Full response: {}".format(result)
            )
        final_waveform = torch.tensor(result["audio"])
        return final_waveform

    @torch.no_grad()
    def generate_speech_from_audio_tensor(self, audio_tensor, padding_mask_audio, text_prompt):
        """
        Generates speech from a preprocessed audio tensor.
        
        Args:
            audio_tensor (Tensor): Preprocessed audio (1D or 2D with channel dim).
            padding_mask_audio (Tensor): Padding mask corresponding to the audio.
            text_prompt (str): Text prompt for conditioning.
        
        Returns:
            Tensor: Generated audio waveform.
        """
        # Encode the audio into latents and scales using NAC's audio_encoder.
        audio_latents, audio_scales = self.nac.audio_encoder(audio_tensor, padding_mask_audio)
        max_length = self.nac.language_model.config.n_positions
        audio_latents = audio_latents[:, :, :max_length].mean(dim=1)
        
        # Tokenize the text prompt and obtain text embeddings.
        text_tokens = self.text_tokenizer(text_prompt, return_tensors="pt").input_ids.to(self.device)
        text_tokens = text_tokens[:, :max_length]
        text_embeddings = self.nac.language_model.transformer.wte(text_tokens)

        # Refine latents using DiTTO reverse diffusion sampling.
        refined_latents = self.__sample_latents(text_embeddings, audio_latents, self.ditto_model, diffusion_steps=ConfigDiTTO.DIFFUSION_STEPS)
        
        # Decode refined latents into final audio.
        final_waveform = self.generate_speech_from_latents(refined_latents, audio_scales, padding_mask_audio)
        return final_waveform

    def generate_speech_from_file(self, file_path, text_prompt):
        """
        Loads a raw audio file, pre-processes it, and generates the final speech waveform.
        
        Args:
            file_path (str): Path to the audio file.
            text_prompt (str): Text prompt for conditioning.
        
        Returns:
            Tensor: Generated audio waveform.
        """
        waveform, sr = torchaudio.load(file_path)
        if sr != self.mel_transform.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.mel_transform.sample_rate)
            waveform = resampler(waveform)
        
        processor_out = self.audio_processor(
            raw_audio=waveform.squeeze().numpy(),
            sampling_rate=self.mel_transform.sample_rate,
            return_tensors="pt",
        )
        processed_audio = processor_out["input_values"].squeeze(0) 
        padding_mask_audio = processor_out["padding_mask"].squeeze(0)  
        
        return self.generate_speech_from_audio_tensor(processed_audio, padding_mask_audio, text_prompt)

    def __p_sample(self, x, t, text_emb):
        """
        Perform one reverse diffusion step using DiTTO.
        
        Args:
            x: [batch, seq_len, hidden_dim] - current latent (x_t).
            t: [batch] - current time step (Long tensor).
            text_emb: [batch, text_seq, text_dim] - text embeddings for conditioning.
        
        Returns:
            x_prev: [batch, seq_len, hidden_dim] - denoised latent (x_{t-1}).
        """
        # Predict the noise using DiTTO.
        noise_pred = self.ditto_model(x, text_emb, t)

        # Retrieve coefficients for the current timestep.
        beta_t = self.ditto_model.betas[t].view(-1, 1, 1)
        alpha_t = self.ditto_model.alphas[t].view(-1, 1, 1)
        alpha_cumprod_t = self.ditto_model.alphas_cumprod[t].view(-1, 1, 1)

        # Compute predicted x0 using the DDPM formulation.
        x0_pred = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)

        # For t > 0, add noise; for t == 0, no noise is added.
        noise = torch.randn_like(x)
        mask = (t > 0).float().view(-1, 1, 1)

        # Reverse process update (DDPM update rule).
        x_prev = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * noise_pred) \
                 + mask * torch.sqrt(beta_t) * noise
        return x_prev

    @torch.no_grad()
    def __sample_latents(self, text_emb, initial_latents, ditto_model, diffusion_steps):
        """
        Sample latents from initial latents using DDPM reverse diffusion.
        
        Args:
            text_emb: [batch, text_seq, text_dim] - text embeddings for conditioning.
            initial_latents: [batch, hidden_dim] - initial latents to refine.
            ditto_model: DiTTO model.
            diffusion_steps (int): Number of diffusion steps.
        
        Returns:
            Refined latents after reverse diffusion sampling.
        """
        device = text_emb.device
        x = initial_latents.to(device)
        # Loop from diffusion_steps - 1 down to 0.
        for t_val in reversed(range(diffusion_steps)):
            t_tensor = torch.full((x.shape[0],), t_val, device=device, dtype=torch.long)
            x = self.__p_sample(x, t_tensor, text_emb)
        return x
