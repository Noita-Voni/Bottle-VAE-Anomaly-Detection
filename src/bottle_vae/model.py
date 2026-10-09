"""Four-block convolutional VAE; defaults retain E1's 128/16 architecture."""

import torch
from torch import nn


class BottleVAE(nn.Module):
    """A small encoder, Gaussian latent distribution, and mirrored decoder."""

    image_size = 128
    latent_dim = 16

    def __init__(self, image_size=128, latent_dim=16):
        super().__init__()
        if image_size not in (128, 256, 512):
            raise ValueError("Implemented image sizes are 128 (E1), 256 (E2) and 512 (E3).")
        if not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError("Latent dimension must be a positive integer.")
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.feature_side = image_size // 16  # Four blocks each halve the side length.
        flattened_features = 128 * self.feature_side * self.feature_side
        # Each stride-2 convolution halves height and width; ReLU adds nonlinearity.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # 128 channels; spatial sides 8 (E1), 16 (E2), 32 (E3).
        )
        self.mean_head = nn.Linear(flattened_features, self.latent_dim)
        self.log_variance_head = nn.Linear(flattened_features, self.latent_dim)
        self.decoder_input = nn.Sequential(
            nn.Linear(self.latent_dim, flattened_features),
            nn.ReLU(),
        )
        # Mirrored transposed convolutions double height and width at each step.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Three independent RGB intensities in [0, 1].
        )

    def encode(self, images):
        """Compress a batch into a mean and log variance for each latent value."""
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(f"Expected shape (batch, 3, {self.image_size}, {self.image_size}), in NCHW order.")
        if images.shape[0] == 0:
            raise ValueError("The VAE needs at least one image in the batch.")
        features = self.encoder(images).flatten(start_dim=1)
        return self.mean_head(features), self.log_variance_head(features)

    def reparameterize(self, mean, log_variance):
        """Sample z while preserving the path for gradients during later training."""
        standard_deviation = torch.exp(0.5 * log_variance)
        noise = torch.randn_like(standard_deviation)
        return mean + standard_deviation * noise

    def decode(self, latent):
        """Expand latent vectors into RGB images at the configured resolution."""
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim or latent.shape[0] == 0:
            raise ValueError(f"Latent vectors must have shape (batch, {self.latent_dim}), with batch > 0.")
        features = self.decoder_input(latent).reshape(-1, 128, self.feature_side, self.feature_side)
        return self.decoder(features)

    def forward(self, images, sample=None):
        """Return reconstruction, mean and log variance; eval uses the mean by default."""
        mean, log_variance = self.encode(images)
        # Training samples; evaluation is repeatable. An explicit flag overrides this.
        if sample is None:
            sample = self.training
        latent = self.reparameterize(mean, log_variance) if sample else mean
        return self.decode(latent), mean, log_variance


# Keep existing E1 imports and state-dictionary keys compatible with saved checkpoints.
E1VAE = BottleVAE
