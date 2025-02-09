import torch
import torch.nn as nn 
import torch.nn.functional as F
from model.NeuralAudioCodec import NAC
from components.DiT import DiT, GlobalAdaLN, RotaryEmbedding

class DiTTO(nn.Module):
    """Full DiT Architecture (Section 3.3)"""

    def __init__(self,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        time_dim=256,
        text_dim=768,
        diffusion_steps=1000,
        lambda_factor=0.1,
        nac_model_path="/tempory/M2-DAC/UE_DEEP/AMAL/DiTTO-TTS/src/params/NAC_epoch_20.pth"
    ):
        super().__init__()

        print("[INFO] Loading NAC model...")
        self.nac = NAC(lambda_factor=lambda_factor)
        nac_info = torch.load(nac_model_path)
        self.nac.load_state_dict(nac_info["model_state_dict"])
        self.nac.eval() 

        for param in self.nac.language_model.parameters():
            param.requires_grad = False

        for param in self.nac.audio_encoder.parameters():
            param.requires_grad = False  
        
        print("[INFO] NAC Loaded.")

        # Time embedding: add an embedding layer for time steps
        self.t_embedding = nn.Embedding(diffusion_steps, time_dim)

        # Time embedding MLP (further processing)
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Global AdaLN (shared across all layers)
        self.ada_ln = GlobalAdaLN(hidden_dim, time_dim, text_dim)

        # DiT Blocks
        self.blocks = nn.ModuleList([
            DiT(hidden_dim, num_heads, time_dim, text_dim)
            for _ in range(num_layers)
        ])

        # Long skip connection
        self.proj_in = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)

        # Rotary embeddings
        self.rotary = RotaryEmbedding(hidden_dim // num_heads)

        # Precompute noise schedule parameters
        self.register_buffer(
            'alphas_cumprod', self.cosine_beta_schedule(diffusion_steps))

    def forward(self, x, text_emb, t):
        """
        Args:
            x: [batch, seq_len, hidden_dim] - noisy latents
            text_emb: [batch, text_seq, text_dim] - text embeddings
            t: [batch] - time indices (Long tensor)
        """
        # Embed time indices
        # now t is of shape [batch, time_dim] and type float
        t = self.t_embedding(t)
        t = self.time_embed(t)

        # Rotary positions
        seq_len = x.shape[1]
        rotary_pos = self.rotary(seq_len, x.device)

        # Long skip connection
        x_skip = self.proj_in(x)

        # Apply global AdaLN modulation
        x = self.ada_ln(x, t, text_emb)

        # Process through DiT blocks
        for block in self.blocks:
            x = block(x, text_emb, t, rotary_pos)

        # Final projection and skip
        x = self.proj_out(x)
        return x_skip + x

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine noise schedule as described in the paper"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(
            ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process (Eqn. 1 in paper)"""
        if noise is None:
            noise = torch.randn_like(x_start)

        # Ensure t is of type long for indexing
        t = t.long()

        # Get alpha cumulative product for each t
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t] ** 0.5
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]) ** 0.5

        # Ensure proper broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.reshape(
            -1, 1, 1)

        return (
            sqrt_alphas_cumprod_t * x_start +
            sqrt_one_minus_alphas_cumprod_t * noise
        )


if __name__ == '__main__':
    # Quick test to check the model (it does :*)
    batch_size = 4
    seq_len = 128
    hidden_dim = 768
    text_dim = 768
    time_dim = 256
    diffusion_steps = 1000

    model = DiTTO(
        hidden_dim=hidden_dim,
        num_layers=12,
        num_heads=12,
        time_dim=time_dim,
        text_dim=text_dim,
        diffusion_steps=diffusion_steps
    )

    x_start = torch.randn(batch_size, seq_len, hidden_dim)
    text_emb = torch.randn(batch_size, seq_len, text_dim)
    t = torch.randint(0, diffusion_steps, (batch_size,)
                      ) 
    noise = torch.randn_like(x_start)

    noisy_latents = model.q_sample(x_start, t, noise)
    predicted_noise = model(noisy_latents, text_emb, t)

    print(predicted_noise.shape)  # Expected: [4, 128, 768]
    loss = F.mse_loss(predicted_noise, noise)
    print(f"Loss: {loss.item()}")
