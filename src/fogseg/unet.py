"""A plain U-Net that turns a stack of scene channels into one fog logit per pixel."""

import torch
from torch import nn
from torch.nn import functional as F


def double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Encoder-decoder with skip connections; the input is zero-padded to a multiple of 2**depth."""

    def __init__(self, in_channels: int, base_channels: int, depth: int):
        super().__init__()
        self.depth = depth
        widths = [base_channels * 2**level for level in range(depth + 1)]
        self.down = nn.ModuleList(
            double_conv(i, o) for i, o in zip([in_channels, *widths[:-2]], widths[:-1], strict=True)
        )
        self.bottom = double_conv(widths[-2], widths[-1])
        self.up = nn.ModuleList(
            double_conv(widths[level + 1] + widths[level], widths[level]) for level in reversed(range(depth))
        )
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        multiple = 2**self.depth
        x = F.pad(x, (0, -width % multiple, 0, -height % multiple))
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottom(x)
        for block, skip in zip(self.up, reversed(skips), strict=True):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([skip, x], dim=1))
        return self.head(x)[:, 0, :height, :width]
