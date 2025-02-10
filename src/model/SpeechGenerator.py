
import torch
import torchaudio
from transformers import AutoProcessor, AutoTokenizer

from model.SpeechLP import SLP
from model.DiTTO import DiTTO
from utils.Config import ConfigDiTTO, ConfigSLP

from tqdm import tqdm 

from bigvgan_v2_24khz_100band_256x import bigvgan
from bigvgan_v2_24khz_100band_256x.meldataset import get_mel_spectrogram



class SpeechGenerator:
    def __init__(
        self,
        lambda_factor,
        nac_model_path="/tempory/M2-DAC/UE_DEEP/AMAL/DiTTO-TTS/src/params/NAC_epoch_20.pth",
        ditto_model_path="/tempory/M2-DAC/UE_DEEP/AMAL/DiTTO-TTS/src/params/DiTTO_epoch_20.pth",
        slp_path = "/tempory/M2-DAC/UE_DEEP/AMAL/DiTTO-TTS/src/params/SLP_epoch_20.pth",
        sample_rate=24000,
        device="cpu"
    ):
        self.device = device

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

        self.slp = SLP(
            ConfigSLP.NB_CLASSES,
            ConfigSLP.NUM_HEADS,
            ConfigSLP.NUM_LAYERS
        )
        with torch.no_grad():
            slp_info = torch.load(slp_path, map_location=self.device)
            self.slp.load_state_dict(slp_info["model_state_dict"])
            self.slp.eval()

        self.sample_rate = sample_rate  
 
        # Initialize audio processor and text tokenizer 
        self.audio_processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.text_tokenizer = AutoTokenizer.from_pretrained("gpt2")

        self.betas = self.ditto_model.cosine_beta_schedule(ConfigDiTTO.DIFFUSION_STEPS).to(self.device)
        self.alphas = (1.0 - self.betas).to(self.device)                     
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)


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

        return self.generate_speech_from_audio_tensor(processed_audio, padding_mask_audio, text_prompt, True)


    @torch.no_grad()
    def generate_speech_from_audio_tensor(self, audio_tensor, padding_mask_audio, text_prompt, is_tokenized=False, is_slp=False):
        # Encode the audio to obtain latents and scales
        audio_latents, audio_scales = self.ditto_model.nac.audio_encoder(audio_tensor, padding_mask_audio)
        max_length = self.ditto_model.nac.language_model.config.n_positions
        audio_latents = audio_latents[:, :, :max_length].mean(dim=1)

        # Tokenize the text prompt and extract text embeddings
        text_tokens = self.text_tokenizer(text_prompt, return_tensors="pt").input_ids.to(self.device) if not is_tokenized else text_prompt
        text_tokens = text_tokens[:, :max_length]
        text_embeddings = self.ditto_model.nac.language_model.transformer.wte(text_tokens)

        # Apply the reverse diffusion process on the latents
        t = torch.full((audio_latents.size(0),), ConfigDiTTO.DIFFUSION_STEPS-1, device=self.device, dtype=torch.long)

        audio_latents = self.ditto_model.q_sample(audio_latents, t)
        refined_latents = self.__sample_latents(text_embeddings, audio_latents, text_tokens, audio_tensor, is_slp)

        return self.__generate_speech_from_latents(refined_latents, audio_scales, padding_mask_audio)
    

    @torch.no_grad()
    def __generate_speech_from_latents(self, audio_latents, audio_scales, padding_mask_audio):
        # Quantize the latents and decode the audio waveform
        audio_latents = audio_latents.unsqueeze(1).repeat(1, 2, 1, 1)
        audio_latents_quantized = self.ditto_model.nac.vector_quantizer(audio_latents)
        waveform = self.ditto_model.nac.audio_decoder.decode(
            audio_latents_quantized.unsqueeze(0).detach(),
            audio_scales=audio_scales,
            padding_mask=padding_mask_audio,
        )[0]
        waveform = waveform.squeeze(1)
        mel_spectrogram = get_mel_spectrogram(waveform, self.vocoder.h).to(self.device)
        generated_waveform = self.vocoder(mel_spectrogram)

        return generated_waveform.squeeze(0)
    
    @torch.no_grad()
    def __p_sample(self, x, t, text_emb, audio_emb=None):
        """
            Reverse diffusion step.
        """

        cond_emb = text_emb if audio_emb is None else torch.cat([text_emb, audio_emb], dim=1)
        noise_pred = self.ditto_model(x, cond_emb, t)
        
        beta_t = self.betas[t].view(-1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)

        noise = torch.randn_like(x)
        mask = (t > 0).float().view(-1, 1, 1)
        x_prev = (1 / torch.sqrt(alpha_t)) * (
                    x - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * noise_pred
                ) + mask * torch.sqrt(beta_t) * noise
        
        return x_prev

    @torch.no_grad()
    def __sample_latents(self, text_emb, audio_emb, text_prompt=None, audio=None, is_slp=False):
        """
            All reverse diffusion steps
        """
        x = torch.randn_like(audio_emb)

        if is_slp:
            L_pred = self.slp(text_prompt, audio)
            x = torch.randn((1, L_pred, audio_emb.shape[-1]), device=self.device)
            
        device = x.device
        for t_val in tqdm(reversed(range(ConfigDiTTO.DIFFUSION_STEPS))):
            t_tensor = torch.full((x.shape[0],), t_val, device=device, dtype=torch.long)
            x = self.__p_sample(x, t_tensor, text_emb, audio_emb)
        return x