"""Direct RGB or bicubic-residual convolutional baselines"""

import torch
from torch import nn


class TinySR(nn.Module):
    """Predict RGB pixels directly, or residuals added to PIL bicubic upsampling."""

    def __init__(self, variant="direct", channels=16, blocks=2, scale=4):
        super().__init__()
        if variant not in ("direct", "residual"):
            raise ValueError("variant must be direct or residual")
        if channels < 4 or blocks < 1 or scale < 2:
            raise ValueError("Invalid network geometry")
        self.variant = variant
        self.channels = int(channels)
        self.blocks = int(blocks)
        self.scale = int(scale)
        layers = [nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(blocks):
            layers.extend(
                [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)]
            )
        layers.extend(
            [nn.Conv2d(channels, 3 * scale * scale, 1), nn.PixelShuffle(scale)]
        )
        self.network = nn.Sequential(*layers)
        # at initialization, residual model exactly equals the bicubic baseline
        if variant == "residual":
            final_conv = self.network[-2]
            nn.init.zeros_(final_conv.weight)
            nn.init.zeros_(final_conv.bias)

    def forward(self, low_res, bicubic=None):
        """Return RGB [B,3,H*scale,W*scale], without output clipping for gradients."""
        prediction = self.network(low_res)
        if self.variant == "residual":
            if bicubic is None or bicubic.shape != prediction.shape:
                raise ValueError(
                    "Residual model requires a matching PIL-bicubic baseline tensor"
                )
            return prediction + bicubic
        return prediction


def model_cost(model, image_size=64):
    """Count dense convolution MACs, weights, and total parameter bytes (FP32)."""
    macs = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # every layer preserves the input spatial dimensions in this model
            macs += (
                image_size**2
                * module.out_channels
                * (module.in_channels // module.groups)
                * module.kernel_size[0]
                * module.kernel_size[1]
            )
    params = sum(p.numel() for p in model.parameters())
    return {
        "macs": int(macs),
        "parameters": int(params),
        "fp32_weight_bytes": int(params * 4),
        "int8_weight_bytes_approx": int(params),
    }
