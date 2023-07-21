import torch.nn as nn
import torch
from torch.nn.functional import relu


class GreenBlock(nn.Module):
    def __init__(self, in_chnnels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_chnnels, in_chnnels,
                               kernel_size=(1, 1), padding="same")
        self.group_norm1 = nn.GroupNorm(num_groups=8, num_channels=in_chnnels)
        self.conv2 = nn.Conv2d(in_chnnels, in_chnnels,
                               kernel_size=(3, 3), padding="same")
        self.group_norm2 = nn.GroupNorm(num_groups=8, num_channels=in_chnnels)

    def forward(self, x):

        inp_res = self.conv1(x)
        x = self.group_norm1(x)
        x = relu(x)

        x = self.conv2(x)
        x = self.group_norm2(x)
        x = relu(x)

        return inp_res + x


class GreenUpSampling(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            assert (in_channels % 2 == 0)
            out_channels = in_channels // 2

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.up_sample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.green_block = GreenBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = relu(x)
        x = self.up_sample(x)
        x = relu(x)
        return self.green_block(x)


class VAEDecoder(nn.Module):
    def __init__(self, intermediate_image_shape=(16, 14, 14), input_size=2048,
                 intermediate_size=128, base_fillter_num=256, output_channel_num=3):
        super().__init__()

        if isinstance(intermediate_image_shape, int):
            intermediate_image_shape = (
                16, intermediate_image_shape // 16, intermediate_image_shape // 16)
        C, H, W = intermediate_image_shape

        self.base_dim = C
        self.base_height = H
        self.base_width = W

        self.mean_linear_layer = nn.Linear(input_size, intermediate_size)
        self.log_var_linear_layer = nn.Linear(input_size, intermediate_size)

        self.linear_layer = nn.Linear(
            intermediate_size, self.base_dim * self.base_height * self.base_width)

        self.conv1 = nn.Conv2d(
            self.base_dim, base_fillter_num, kernel_size=(1, 1))
        self.up_samapling1 = nn.ConvTranspose2d(
            base_fillter_num, base_fillter_num, kernel_size=2, stride=2)

        self.green_upsampaling1 = GreenUpSampling(base_fillter_num)
        self.green_upsampaling2 = GreenUpSampling(base_fillter_num // 2)
        self.green_upsampaling3 = GreenUpSampling(base_fillter_num // 4)

        self.conv_ending = nn.Conv2d(
            base_fillter_num // 8, output_channel_num, kernel_size=(1, 1))

    def sample(self, mean, var):
        # assert(mean.device == var.device)
        return mean + (var ** 0.5) * torch.randn(mean.shape, device=mean.device)

    def decode(self, x):
        x = self.linear_layer(x)
        x = relu(x)
        x = x.view(-1, self.base_dim, self.base_height, self.base_width)

        x = self.conv1(x)
        x = self.up_samapling1(x)

        x = self.green_upsampaling1(x)
        x = self.green_upsampaling2(x)
        x = self.green_upsampaling3(x)

        x = self.conv_ending(x)

        return x

    def forward(self, x):

        mean = self.mean_linear_layer(x)
        z_log_sigma2 = self.log_var_linear_layer(x)
        var = torch.exp(z_log_sigma2)

        sample = self.sample(mean, var)

        decoded = self.decode(sample)

        return decoded, mean, z_log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None

    dx = (torch.numel(x) // x.shape[0])
    data_loss = ((x - xr) ** 2).sum(dim=(1, 2, 3)) / (x_sigma2 * dx)

    dz = z_mu.shape[1]

    z_sigma2 = torch.exp(z_log_sigma2)

    tr_z = z_sigma2.sum(axis=1)

    kldiv_loss = tr_z + (z_mu ** 2).sum(axis=1) - dz - z_log_sigma2.sum(axis=1)

    data_loss = data_loss.mean()
    kldiv_loss = kldiv_loss.mean()

    loss = data_loss + kldiv_loss

    return loss
