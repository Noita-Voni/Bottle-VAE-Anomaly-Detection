"""VAE objective: Gaussian reconstruction loss plus latent KL divergence."""


def vae_loss(reconstruction, images, mean, log_variance):
    """Return total, reconstruction and KL losses, each averaged across the batch.

    Sum the squared RGB errors per image and multiply by 0.5. This corresponds
    to a Gaussian observation model with fixed variance 1, omitting its constant.
    Sum the latent KL terms per image against a standard normal prior.
    """
    if images.ndim != 4 or reconstruction.shape != images.shape or images.shape[0] == 0:
        raise ValueError("Images and reconstructions must have the same non-empty NCHW shape.")
    if (mean.ndim != 2 or log_variance.shape != mean.shape
            or mean.shape[0] != images.shape[0] or mean.shape[1] == 0):
        raise ValueError("Mean and log variance must have matching (batch, latent) shapes.")

    # Sum over channels and pixels first, then average across images.
    reconstruction_per_image = 0.5 * (reconstruction - images).square().flatten(1).sum(1)
    reconstruction_loss = reconstruction_per_image.mean()

    # Closed-form KL between the encoded diagonal Gaussian and N(0, I).
    kl_per_image = -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).sum(1)
    kl_loss = kl_per_image.mean()

    total_loss = reconstruction_loss + kl_loss
    return total_loss, reconstruction_loss, kl_loss
