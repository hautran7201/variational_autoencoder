import torch 
import torch.nn as nn 
from torch.nn import functional as F
from typing import Optional, TypeVar


_T = TypeVar('_T', bound=nn.Module)

def zero_module(module: _T) -> _T:
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResNetBlock(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: Optional[int] = None,
        use_conv_shortcut: bool = False,
        dropout_probability: float = 0.0,
        zero_init_last: bool = False,
    ):
        super().__init__()

        self.input_channel = input_channel
        self.output_channel = output_channel if output_channel is not None else input_channel
        self.use_conv_shortcut = use_conv_shortcut
        self.dropout_probability = dropout_probability
        self.zero_init_last = zero_init_last 

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=self.input_channel, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(self.input_channel, self.output_channel, kernel_size=3, stride=1, padding=1)
        
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='linear')
        self.conv1.weight.data *= 1.6761
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.output_channel, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(p=self.dropout_probability)
        self.conv2 = nn.Conv2d(self.output_channel, self.output_channel, kernel_size=3, stride=1, padding=1)

        if input_channel != output_channel:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(self.input_channel, self.output_channel, kernel_size=3, stride=1, padding=1)
            else:
                self.conv_shortcut = nn.Conv2d(self.input_channel, self.output_channel, kernel_size=1, stride=1, padding=0)
            nn.init.kaiming_normal_(self.conv_shortcut.weight, nonlinearity='linear')
        else:
            self.conv_shortcut = nn.Identity()

        if self.zero_init_last:
            self.residual_scale = 1.0 
            self.conv2 = zero_module(self.conv2)
        else:
            self.residual_scale = 0.70711
            nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='linear')
            self.conv2.weight.data *= 1.6761 * self.residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.residual_scale * self.conv_shortcut(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + shortcut


class AttentionLayer(nn.Module):
    def __init__(self, input_channel, dropout_probability: float=0.0):
        super().__init__()

        self.input_channel = input_channel 
        self.dropout_probability = dropout_probability

        self.qkv_norm = nn.GroupNorm(32, self.input_channel, eps=1e-6, affine=True)
        self.qkv = nn.Conv2d(self.input_channel, 3*self.input_channel, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.qkv.weight, nonlinearity='linear')

        # self.proj_norm = nn.GroupNorm(32, self.input_channel, eps=1e-6, affine=True)
        self.proj = nn.Conv2d(self.input_channel, self.input_channel, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity='linear')

    def _reshape_for_attention(self, x):
        # x: (batch, channel, height, width)
        # (batch, channel, height, width) -> (batch, height, width, channel) -> (batch, height * width, channel)
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1]).contiguous()
        return x

    def _reshape_from_attention(self, x, height, width):
        # x: (batch, height * width, channel)
        # (batch, height * width, channel) -> (batch, height, width, channel) -> (batch, channel, height, width)
        x = x.reshape(x.shape[0], height, width, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x):
        # x: (batch, channel, height,  width)
        height, width = x.shape[-2:]
        # residual = x

        # (batch, channel, height,  width) -> (batch, channel, height,  width)
        h = self.qkv_norm(x)
        # (batch, channel, height,  width) -> (batch, 3*channel, height,  width)
        qkv = self.qkv(h)

        # (batch, 3*channel, height,  width) -> (batch, height * width, 3*channel)
        qkv = self._reshape_for_attention(qkv)
        # (batch, height * width, 3*channel) -> 3*(batch, height * width, channel)
        q, k, v = torch.split(qkv, self.input_channel, dim=2)

        # 3*(batch, height * width, channel) -> (batch, height * width, channel)
        h = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_probability)

        # (batch, height * width, channel) -> (batch, channel, height, width)
        h = self._reshape_from_attention(h, height, width) # + residual
        # (batch, channel, height, width)
        # residual = h

        # (batch, channel, height, width) -> (batch, channel, height, width)
        h = self.proj(h)

        return h + x


class Downsample(nn.Module):
    def __init__(self, input_channel, resample_with_conv):
        super().__init__()

        self.input_channel = input_channel 
        self.resample_with_conv = resample_with_conv

        if self.resample_with_conv:
            self.conv = nn.Conv2d(self.input_channel, self.input_channel, kernel_size=3, stride=2, padding=0)
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear')
    
    def forward(self, x):
        if self.resample_with_conv:
            h = F.pad(x, (0, 1, 0, 1), mode='constant', value=0)
            h = self.conv(h)
        else:
            h = F.avg_pool2d(x, kernel_size=2, stride=2)
        return h


class Upsample(nn.Module):
    def __init__(self, input_channel, resample_with_conv):
        super().__init__()
        self.input_channel = input_channel
        self.resample_with_conv = resample_with_conv 

        if self.resample_with_conv:
            self.conv = nn.Conv2d(self.input_channel, self.input_channel, kernel_size=3, stride=1, padding=1)
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear')

    def forward(self, x):
        h = F.interpolate(x, scale_factor=2, mode='nearest', antialias=False)        
        if self.resample_with_conv:
            h = self.conv(h)
        return h


class GradientScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 1.0

    def set_scale(self, scale: float):
        self.scale = scale

    def forward(self, x):
        return x

    def backward_hook(self, module, grad_input, grad_output):
        return (self.scale * grad_input[0])        