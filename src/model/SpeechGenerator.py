import os
import sys
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import AutoProcessor, AutoTokenizer, AutoModel

from model.NeuralAudioCodec import NAC
from model.DiTTO import DiTTO
from utils.Config import ConfigDiTTO

from bigvgan_v2_24khz_100band_256x import bigvgan
from bigvgan_v2_24khz_100band_256x.meldataset import get_mel_spectrogram



class SpeechGenerator:
    def __init__(
        self,
        lambda_factor,
        nac_model_path="/tempory/M2-DAC/UE_DEEP/AMAL/DiTTO-TTS/src/params/NAC_epoch_20.pth",
        ditto_model_path="/tempory/M2-DAC/UE_DEEP/AMAL/DiTTO-TTS/src/params/DiTTO_epoch_20.pth",
        sample_rate=24000,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the DiTTO model
        self.ditto_model = DiTTO(
            hidden_dim=ConfigDiTTO.HIDDEN_DIM,
            num_layers=ConfigDiTTO.NUM_LAYERS,
            num_heads=ConfigDiTTO.NUM_HEADS,
            time_dim=ConfigDiTTO.TIME_DIM,
            text_dim=ConfigDiTTO.TEXT_EMBED_DIM,
            diffusion_steps=ConfigDiTTO.DIFFUSION_STEPS,
            lambda_factor=lambda_factor,
            nac_model_path=nac_model_path,
        ).to(self.device)

        with torch.no_grad():
            ditto_info = torch.load(ditto_model_path, map_location=self.device)
            self.ditto_model.load_state_dict(ditto_info["model_state_dict"])
            self.ditto_model.eval()
        
        # Initialize the BigVGAN vocoder from the pretrained checkpoint
        self.vocoder = bigvgan.BigVGAN.from_pretrained(
            'nvidia/bigvgan_v2_24khz_100band_256x',
            use_cuda_kernel=False
        )
        self.vocoder.remove_weight_norm()
        self.vocoder = self.vocoder.eval().to(self.device)

        self.sample_rate = sample_rate  
 
        # Initialize audio processor and text tokenizer 
        self.audio_processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.text_tokenizer = AutoTokenizer.from_pretrained("gpt2")


    def generate_speech_from_file(self, file_path, text_prompt):
        # Load waveform from file and resample if necessary
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

        # Process the audio using the encoder's processor
        processor_out = self.audio_processor(
            raw_audio=waveform.squeeze().numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        processed_audio = processor_out["input_values"].squeeze(0)
        padding_mask_audio = processor_out.get("padding_mask", torch.ones_like(processed_audio))

        return self.generate_speech_from_audio_tensor(processed_audio, padding_mask_audio, text_prompt)


    @torch.no_grad()
    def generate_speech_from_audio_tensor(self, audio_tensor, padding_mask_audio, text_prompt):
        # Encode the audio to obtain latents and scales
        audio_latents, audio_scales = self.ditto_model.nac.audio_encoder(audio_tensor, padding_mask_audio)
        max_length = self.ditto_model.nac.language_model.config.n_positions
        audio_latents = audio_latents[:, :, :max_length].mean(dim=1)

        # Tokenize the text prompt and extract text embeddings
        text_tokens = self.text_tokenizer(text_prompt, return_tensors="pt").input_ids.to(self.device)
        text_tokens = text_tokens[:, :max_length]
        text_embeddings = self.ditto_model.nac.language_model.transformer.wte(text_tokens)

        # Apply the reverse diffusion process on the latents
        audio_latents = self.ditto_model.q_sample(audio_latents, ConfigDiTTO.DIFFUSION_STEPS)
        refined_latents = self.__sample_latents(text_embeddings, audio_latents)

        return self.__generate_speech_from_latents(refined_latents, audio_scales, padding_mask_audio)
    

    @torch.no_grad()
    def __generate_speech_from_latents(self, audio_latents, audio_scales, padding_mask_audio):
        # Quantize the latents and decode the audio waveform
        audio_latents_quantized = self.ditto_model.vector_quantizer(audio_latents)
        waveform = self.ditto_model.nac.audio_decoder.decode(
            audio_latents_quantized.unsqueeze(0).detach(),
            audio_scales=audio_scales,
            padding_mask=padding_mask_audio,
        )[0]

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel_spectrogram = get_mel_spectrogram(waveform, self.vocoder.h).to(self.device)
        generated_waveform = self.vocoder(mel_spectrogram)

        return generated_waveform.squeeze(0)
    

    @torch.no_grad()
    def __p_sample(self, x, t, text_emb):
        # Predict noise using the DiTTO model
        noise_pred = self.ditto_model(x, text_emb, t)
        beta_t = self.ditto_model.betas[t].view(-1, 1, 1)
        alpha_t = self.ditto_model.alphas[t].view(-1, 1, 1)
        alpha_cumprod_t = self.ditto_model.alphas_cumprod[t].view(-1, 1, 1)

        noise = torch.randn_like(x)
        mask = (t > 0).float().view(-1, 1, 1)
        x_prev = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * noise_pred) \
                 + mask * torch.sqrt(beta_t) * noise
        
        return x_prev

    @torch.no_grad()
    def __sample_latents(self, text_emb, initial_latents):
        # Iteratively apply reverse diffusion to refine the latent representation
        x = initial_latents.to(self.device)
        for t_val in reversed(range(ConfigDiTTO.DIFFUSION_STEPS)):
            t_tensor = torch.full((x.shape[0],), t_val, device=self.device, dtype=torch.long)
            x = self.__p_sample(x, t_tensor, text_emb)
        return x
