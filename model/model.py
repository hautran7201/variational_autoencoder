import logging
import math
from typing import List, Optional, Tuple, Union

import torch
from composer.devices import DeviceGPU
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler, UNet2DConditionModel
from torchmetrics import MeanSquaredError
from transformers import CLIPTextModel, CLIPTokenizer, PretrainedConfig

from model.autoencoder import (AutoEncoder, AutoEncoderLoss, ComposerAutoEncoder,
                                          ComposerDiffusersAutoEncoder) # load_autoencoder                                          
# from model.layers import ClippedAttnProcessor2_0, ClippedXFormersAttnProcessor, zero_module
# from model.pixel_diffusion import PixelDiffusion
# from model.stable_diffusion import StableDiffusion
# from model.text_encoder import MultiTextEncoder, MultiTokenizer
# from schedulers.schedulers import ContinuousTimeScheduler


def build_autoencoder(input_channels: int = 3,
                      output_channels: int = 3,
                      hidden_channels: int = 128,
                      latent_channels: int = 4,
                      double_latent_channels: bool = True,
                      channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
                      num_residual_blocks: int = 2,
                      use_conv_shortcut: bool = False,
                      dropout_probability: float = 0.0,
                      resample_with_conv: bool = True,
                      zero_init_last: bool = False,
                      use_attention: bool = True,
                      input_key: str = 'image',
                      learn_log_var: bool = True,
                      log_var_init: float = 0.0,
                      kl_divergence_weight: float = 1.0,
                      lpips_weight: float = 0.25,
                      discriminator_weight: float = 0.5,
                      discriminator_num_filters: int = 64,
                      discriminator_num_layers: int = 3):
    """Autoencoder training setup. By default, this config matches the network architecure used in SD2 and SDXL.

    Args:
        input_channels (int): Number of input channels. Default: `3`.
        output_channels (int): Number of output channels. Default: `3`.
        hidden_channels (int): Number of hidden channels. Default: `128`.
        latent_channels (int): Number of latent channels. Default: `4`.
        double_latent_channels (bool): Whether to double the number of latent channels in the decoder. Default: `True`.
        channel_multipliers (tuple): Tuple of channel multipliers for each layer in the encoder and decoder. Default: `(1, 2, 4, 4)`.
        num_residual_blocks (int): Number of residual blocks in the encoder and decoder. Default: `2`.
        use_conv_shortcut (bool): Whether to use a convolutional shortcut in the residual blocks. Default: `False`.
        dropout_probability (float): Dropout probability. Default: `0.0`.
        resample_with_conv (bool): Whether to use a convolutional resampling layer. Default: `True`.
        zero_init_last (bool): Whether to zero initialize the last layer in resblocks+discriminator. Default: `False`.
        use_attention (bool): Whether to use attention in the encoder and decoder. Default: `True`.
        input_key (str): Key to use for the input. Default: `image`.
        learn_log_var (bool): Whether to learn the output log variance in the VAE. Default: `True`.
        log_var_init (float): Initial value for the output log variance. Default: `0.0`.
        kl_divergence_weight (float): Weight for the KL divergence loss. Default: `1.0`.
        lpips_weight (float): Weight for the LPIPS loss. Default: `0.25`.
        discriminator_weight (float): Weight for the discriminator loss. Default: `0.5`.
        discriminator_num_filters (int): Number of filters in the discriminator. Default: `64`.
        discriminator_num_layers (int): Number of layers in the discriminator. Default: `3`.
    """
    # Build the autoencoder
    autoencoder = AutoEncoder(
        input_channels=input_channels,
        output_channels=output_channels,
        hidden_channels=hidden_channels,
        latent_channels=latent_channels,
        double_latent_channels=double_latent_channels,
        channel_multipliers=channel_multipliers,
        num_residual_blocks=num_residual_blocks,
        use_conv_shortcut=use_conv_shortcut,
        dropout_probability=dropout_probability,
        resample_with_conv=resample_with_conv,
        zero_init_last=zero_init_last,
        use_attention=use_attention,
    )

    # Configure the loss function
    autoencoder_loss = AutoEncoderLoss(input_key=input_key,
                                       ae_output_channels=output_channels,
                                       learn_log_var=learn_log_var,
                                       log_var_init=log_var_init,
                                       kl_divergence_weight=kl_divergence_weight,
                                       lpips_weight=lpips_weight,
                                       discriminator_weight=discriminator_weight,
                                       discriminator_num_filters=discriminator_num_filters,
                                       discriminator_num_layers=discriminator_num_layers)

    composer_model = ComposerAutoEncoder(model=autoencoder, autoencoder_loss=autoencoder_loss, input_key=input_key)
    return composer_model


def build_diffusers_autoencoder(model_name: str = 'stabilityai/stable-diffusion-2-base',
                                pretrained: bool = True,
                                vae_subfolder: bool = True,
                                output_channels: int = 3,
                                input_key: str = 'image',
                                learn_log_var: bool = True,
                                log_var_init: float = 0.0,
                                kl_divergence_weight: float = 1.0,
                                lpips_weight: float = 0.25,
                                discriminator_weight: float = 0.5,
                                discriminator_num_filters: int = 64,
                                discriminator_num_layers: int = 3,
                                zero_init_last: bool = False):
    """Diffusers autoencoder training setup.

    Args:
        model_name (str): Name of the Huggingface model. Default: `stabilityai/stable-diffusion-2-base`.
        pretrained (bool): Whether to use a pretrained model. Default: `True`.
        vae_subfolder: (bool): Whether to find the model config in a vae subfolder. Default: `True`.
        output_channels (int): Number of output channels. Default: `3`.
        input_key (str): Key for the input to the model. Default: `image`.
        learn_log_var (bool): Whether to learn the output log variance. Default: `True`.
        log_var_init (float): Initial value for the output log variance. Default: `0.0`.
        kl_divergence_weight (float): Weight for the KL divergence loss. Default: `1.0`.
        lpips_weight (float): Weight for the LPIPs loss. Default: `0.25`.
        discriminator_weight (float): Weight for the discriminator loss. Default: `0.5`.
        discriminator_num_filters (int): Number of filters in the first layer of the discriminator. Default: `64`.
        discriminator_num_layers (int): Number of layers in the discriminator. Default: `3`.
        zero_init_last (bool): Whether to initialize the last conv layer to zero. Default: `False`.
    """
    # Get the model architecture and optionally the pretrained weights.
    if pretrained:
        if vae_subfolder:
            model = AutoencoderKL.from_pretrained(model_name, subfolder='vae')
        else:
            model = AutoencoderKL.from_pretrained(model_name)
    else:
        if vae_subfolder:
            config = PretrainedConfig.get_config_dict(model_name, subfolder='vae')
        else:
            config = PretrainedConfig.get_config_dict(model_name)
        model = AutoencoderKL(**config[0])
    assert isinstance(model, AutoencoderKL)

    # Configure the loss function
    autoencoder_loss = AutoEncoderLoss(input_key=input_key,
                                       ae_output_channels=output_channels,
                                       learn_log_var=learn_log_var,
                                       log_var_init=log_var_init,
                                       kl_divergence_weight=kl_divergence_weight,
                                       lpips_weight=lpips_weight,
                                       discriminator_weight=discriminator_weight,
                                       discriminator_num_filters=discriminator_num_filters,
                                       discriminator_num_layers=discriminator_num_layers)

    # Make the composer model
    composer_model = ComposerDiffusersAutoEncoder(model=model, autoencoder_loss=autoencoder_loss, input_key=input_key)
    return composer_model