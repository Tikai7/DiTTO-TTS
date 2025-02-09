import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class GlobalAdaLN(nn.Module):
    """Global Adaptive Layer Norm (Section 3.3)"""

    def __init__(self, hidden_dim, time_dim, text_dim):
        super().__init__()
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 2 * hidden_dim)
        )
        # Text embedding projection (mean pooled)
        self.text_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(text_dim, 2 * hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, x, time_emb, text_emb):
        # text_emb: [batch, seq_len, text_dim] → [batch, text_dim]
        text_emb = torch.mean(text_emb, dim=1)

        # Get scale/shift from both embeddings
        time_scale, time_shift = self.time_mlp(time_emb).chunk(2, dim=-1)
        text_scale, text_shift = self.text_mlp(text_emb).chunk(2, dim=-1)

        # Combine modulations (Eqn. in Section 3.3)
        scale = 1 + time_scale + text_scale
        shift = time_shift + text_shift

        # Apply global modulation
        x = self.norm(x)
        x = x * scale.unsqueeze(1) + shift.unsqueeze(1)
        return x


class RotaryEmbedding(nn.Module):
    """RoPE (Section 3.3)"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

    def apply_rope(self, pos, t):
        """
        Args:
            pos: positional encoding of shape [seq_len, head_dim]
            t: tensor of shape [batch, seq_len, num_heads, head_dim]
        Returns:
            Tensor with RoPE applied.
        """
        # Explicitly reshape pos to broadcast over batch and num_heads
        # now shape: [1, seq_len, 1, head_dim]
        pos = pos.unsqueeze(0).unsqueeze(2)
        return t * pos.cos() + self._rotate_half(t) * pos.sin()


class DiTBlock(nn.Module):
    """Single DiT Block (Figure 1-left)"""

    def __init__(self, hidden_dim, num_heads, time_dim, text_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # head dimension

        # Self-attention components
        self.norm1 = nn.LayerNorm(hidden_dim)
        # We still create an instance of MultiheadAttention to hold the projection weights,
        # but we will use its in_proj_weight and in_proj_bias manually.
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.rotary = RotaryEmbedding(self.head_dim)

        # Cross-attention components
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1)

        # Gated MLP (Section 3.3)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.mlp_fc1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.act = nn.GELU()
        self.gate = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.mlp_fc2 = nn.Linear(4 * hidden_dim, hidden_dim)

    def forward(self, x, text_emb, time_emb, rotary_pos):
        batch_size, seq_len, _ = x.shape

        # ===== Self-attention with RoPE =====
        residual = x
        x = self.norm1(x)

        # -- Manually compute q, k, v using in_proj weights --
        d_model = x.size(-1)  # should be hidden_dim
        w = self.attn.in_proj_weight  # shape: [3*d_model, d_model]
        b = self.attn.in_proj_bias    # shape: [3*d_model]
        q = F.linear(x, w[:d_model, :], b[:d_model])
        k = F.linear(x, w[d_model:2*d_model, :], b[d_model:2*d_model])
        v = F.linear(x, w[2*d_model:, :], b[2*d_model:])

        # -- Reshape q, k, v to [batch, seq_len, num_heads, head_dim] --
        q = rearrange(q, 'b n (h d) -> b n h d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b n h d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b n h d', h=self.num_heads)

        # -- Apply RoPE to q and k.
        #    rotary_pos is computed as: [seq_len, head_dim] from RotaryEmbedding.forward().
        q = self.rotary.apply_rope(rotary_pos, q)  # output shape: [b, n, h, d]
        k = self.rotary.apply_rope(rotary_pos, k)

        # -- Permute to [batch, num_heads, seq_len, head_dim] for attention computation --
        q = q.permute(0, 2, 1, 3)  # [b, h, n, d]
        k = k.permute(0, 2, 1, 3)  # [b, h, n, d]
        v = v.permute(0, 2, 1, 3)  # [b, h, n, d]

        # -- Scaled Dot-Product Attention --
        scores = torch.matmul(q, k.transpose(-2, -1)) / \
            math.sqrt(self.head_dim)  # [b, h, n, n]
        attn = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v)  # [b, h, n, d]

        # -- Merge heads back: reshape to [batch, seq_len, hidden_dim] --
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, d_model)
        x = attn_out + residual

        # ===== Cross-attention =====
        residual = x
        x = self.norm2(x)
        # For cross-attention, we use the built-in module which expects [seq_len, batch, embed_dim]
        x = self.cross_attn(
            x.transpose(0, 1),
            text_emb.transpose(0, 1),
            text_emb.transpose(0, 1)
        )[0].transpose(0, 1) + residual

        # ===== Gated MLP =====
        residual = x
        x_norm = self.norm3(x)
        # shape: [batch, seq_len, 4*hidden_dim]
        x_proj = self.act(self.mlp_fc1(x_norm))
        # shape: [batch, seq_len, 4*hidden_dim]
        gate = torch.sigmoid(self.gate(x_norm))
        # shape: [batch, seq_len, hidden_dim]
        x = self.mlp_fc2(x_proj * gate) + residual

        return x


class DiT(nn.Module):
    """Full DiT Architecture (Section 3.3)"""

    def __init__(self,
                 hidden_dim=768,
                 num_layers=12,
                 num_heads=12,
                 time_dim=256,
                 text_dim=768,
                 diffusion_steps=1000):
        super().__init__()

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
            DiTBlock(hidden_dim, num_heads, time_dim, text_dim)
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

    model = DiT(
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
