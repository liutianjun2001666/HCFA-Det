import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HCFDA(nn.Module):
    """
    Physically Inspired Heat Conduction Frequency Domain Attention (HCFDA) module
    with improved heat conduction simulation based on physical laws.

    Args:
        channels (int): Number of input/output channels.
        reduction_ratio (int, optional): Channel reduction ratio. Default: 16.
        thermal_diffusivity (bool, optional): Whether to use learnable thermal diffusivity. Default: True.
        time_steps (int, optional): Number of time steps for heat conduction simulation. Default: 3.
    """

    def __init__(self, channels, reduction_ratio=16, thermal_diffusivity=True, time_steps=3,trainable_laplacian=True):
        super(HCFDA, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.time_steps = time_steps

        # Channel attention components
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )

        # Frequency domain transformation
        self.dct_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # Physical parameters for heat conduction
        if thermal_diffusivity:
            # Thermal diffusivity (α in heat equation ∂u/∂t = α∇²u)
            self.alpha = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_buffer('alpha', torch.tensor(0.1))

        # Laplacian kernel for heat conduction simulation
        if trainable_laplacian:
            self.laplacian_kernel = nn.Parameter(
                torch.tensor([
                    [0.05, 0.2, 0.05],
                    [0.2, -1.0, 0.2],
                    [0.05, 0.2, 0.05]
                ], dtype=torch.float32).view(1, 1, 3, 3)
            )
        else:
            self.register_buffer('laplacian_kernel', torch.tensor([
                [0.05, 0.2, 0.05],
                [0.2, -1.0, 0.2],
                [0.05, 0.2, 0.05]
            ], dtype=torch.float32).view(1, 1, 3, 3))

        self._init_weights()

    def _init_weights(self):
        # Initialize DCT weights
        dct_weight = self._get_dct_filter(self.channels, self.channels)
        self.dct_conv.weight.data = torch.from_numpy(dct_weight).float().unsqueeze(-1).unsqueeze(-1)
        self.dct_conv.weight.requires_grad = False

        # Initialize FC layers
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)

    def _get_dct_filter(self, tile_size_x, tile_size_y):
        """
        Generate DCT filter weights for frequency transformation.
        """
        dct_filter = np.zeros((tile_size_x, tile_size_y))

        for k in range(tile_size_x):
            for l in range(tile_size_y):
                if k == 0 and l == 0:
                    dct_filter[k, l] = np.sqrt(1.0 / tile_size_x) * np.sqrt(1.0 / tile_size_y)
                elif k == 0:
                    dct_filter[k, l] = np.sqrt(1.0 / tile_size_x) * np.sqrt(2.0 / tile_size_y) * np.cos(
                        (2 * l + 1) * k * np.pi / (2 * tile_size_y))
                elif l == 0:
                    dct_filter[k, l] = np.sqrt(2.0 / tile_size_x) * np.sqrt(1.0 / tile_size_y) * np.cos(
                        (2 * k + 1) * l * np.pi / (2 * tile_size_x))
                else:
                    dct_filter[k, l] = np.sqrt(2.0 / tile_size_x) * np.sqrt(2.0 / tile_size_y) * np.cos(
                        (2 * k + 1) * l * np.pi / (2 * tile_size_x)) * np.cos(
                        (2 * l + 1) * k * np.pi / (2 * tile_size_y))

        return dct_filter

    def _apply_heat_conduction(self, x):
        """
        Improved heat conduction simulation based on physical heat equation.
        Uses explicit finite difference method to solve ∂u/∂t = α∇²u.
        """
        # DCT transformation to frequency domain
        x_dct = self.dct_conv(x)

        # Initial temperature distribution (using channel mean)
        temperature = torch.mean(x_dct, dim=1, keepdim=True)  # [B, 1, H, W]

        # Pad for laplacian convolution
        pad = self.laplacian_kernel.size(2) // 2
        padded_temp = F.pad(temperature, [pad] * 4, mode='reflect')

        # Heat conduction simulation over multiple time steps
        for _ in range(self.time_steps):
            # Compute laplacian (∇²u)
            laplacian = F.conv2d(
                padded_temp,
                self.laplacian_kernel,
                padding=0
            )

            # Update temperature using explicit Euler method: u_{t+1} = u_t + α∇²u
            delta = self.alpha * laplacian
            temperature = temperature + delta

            # Apply boundary conditions (reflect padding)
            padded_temp = F.pad(temperature, [pad] * 4, mode='reflect')

        # Normalize the final temperature map
        heat_map = torch.sigmoid(temperature)
        return heat_map

    def forward(self, x):
        b, c, h, w = x.size()

        # Physical heat conduction in frequency domain
        heat_map = self._apply_heat_conduction(x)

        # Channel attention
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        y_avg = self.fc(y_avg).view(b, c, 1, 1)
        y_max = self.fc(y_max).view(b, c, 1, 1)
        channel_att = y_avg + y_max

        # Combine attention mechanisms
        attention = torch.sigmoid(channel_att * heat_map)
        return x * attention.expand_as(x)
