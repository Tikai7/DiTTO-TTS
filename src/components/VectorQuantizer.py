import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):
    """
    Convert latents into discrete indices using a codebook.
    """
    def __init__(self, codebook_size, latent_dim):
        """
        Args:
            codebook_size (int): Number of vectors in the codebook.
            latent_dim (int): Dimension of each latent vector.
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim

        # Initialize the codebook with learnable parameters
        self.codebook = nn.Parameter(torch.randn(codebook_size, latent_dim))
        nn.init.xavier_uniform_(self.codebook)

    def forward(self, latents):
        """
        Args:
            latents (torch.Tensor): Continuous latent vectors of shape [batch_size, channels, num_frames, latent_dim].
        
        Returns:
            indices (torch.LongTensor): Discrete indices of shape [batch_size, channels, num_frames].
        """
        batch_size, num_channels, num_frames, latent_dim = latents.shape
        latents_flattened = latents.view(-1, latent_dim)

        # Compute distances between latents and codebook vectors
        distances = (
            torch.sum(latents_flattened**2, dim=1, keepdim=True) 
            - 2 * torch.matmul(latents_flattened, self.codebook.T)  
            + torch.sum(self.codebook**2, dim=1) 
        )  # Final shape: [B * C * T, K]

        indices = torch.argmin(distances, dim=-1)  # [B * C * T]
        indices = indices.view(batch_size, num_channels, num_frames)

        return indices
