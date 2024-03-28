import sys
sys.path.append('model')

import pytest
import torch
from layers import ResNetBlock, AttentionLayer, Downsample, Upsample
from autoencoder import Encoder, Decoder, AutoEncoder
from diffusers import AutoencoderKL
from transformers import PretrainedConfig


@pytest.mark.parametrize('input_channel', [32])
@pytest.mark.parametrize('output_channel', [32, 64])
@pytest.mark.parametrize('use_conv_shortcut', [False, True])
@pytest.mark.parametrize('dropout_probability', [0.0, 0.5])
@pytest.mark.parametrize('zero_init_last', [False, True])
def test_ResNetBlock(
        input_channel,
        output_channel,
        use_conv_shortcut,
        dropout_probability,
        zero_init_last
    ):

    block = ResNetBlock(
      input_channel=input_channel,
      output_channel=output_channel,
      use_conv_shortcut=use_conv_shortcut,
      dropout_probability=dropout_probability,
      zero_init_last=zero_init_last
    )

    x = torch.rand(1, input_channel, 5, 5)
    y = block(x)

    assert y.shape == (1, output_channel, 5, 5), f'{y.shape}'
    if input_channel == output_channel:
        print(x.shape, y.shape)
        torch.testing.assert_close(x.shape, y.shape)


@pytest.mark.parametrize('input_channel', [32, 64])
@pytest.mark.parametrize('size', [6, 7])
def test_attention(input_channel, size):
    attention = AttentionLayer(input_channel=input_channel)
    x = torch.randn(1, input_channel, size, size)
    y = attention(x)
    assert y.shape == x.shape


@pytest.mark.parametrize('input_channel', [3, 4])
@pytest.mark.parametrize('size', [6, 7])
@pytest.mark.parametrize('resample_with_conv', [True, False])
def test_downsample(input_channel, resample_with_conv, size):
    downsample = Downsample(input_channel=input_channel, resample_with_conv=resample_with_conv)
    x = torch.randn(1, input_channel, size, size)
    y = downsample(x)
    assert y.shape == (1, input_channel, size // 2, size // 2)


@pytest.mark.parametrize('input_channel', [3, 4])
@pytest.mark.parametrize('size', [6, 7])
@pytest.mark.parametrize('resample_with_conv', [True, False])
def test_upsample(input_channel, resample_with_conv, size):
    upsample = Upsample(input_channel=input_channel, resample_with_conv=resample_with_conv)
    x = torch.randn(1, input_channel, size, size)
    y = upsample(x)
    assert y.shape == (1, input_channel, size * 2, size * 2)    


@pytest.mark.parametrize('input_channel', [3])
@pytest.mark.parametrize('hidden_channel', [32])
@pytest.mark.parametrize('latent_dim', [4])
@pytest.mark.parametrize('channel_multiplier', [(1, 2, 4, 8)])
@pytest.mark.parametrize('num_residual_block', [4])
@pytest.mark.parametrize('double_latent_dim', [True, False])
@pytest.mark.parametrize('resample_with_conv', [True, False])
@pytest.mark.parametrize('use_conv_shortcut', [True, False])
@pytest.mark.parametrize('dropout_probability', [0.0])
@pytest.mark.parametrize('zero_init_last', [ True, False])
@pytest.mark.parametrize('use_attention', [True, False])
def test_encoder(
    input_channel,
    hidden_channel,
    latent_dim,
    channel_multiplier,
    num_residual_block,
    double_latent_dim, 
    resample_with_conv,
    use_conv_shortcut,
    dropout_probability,
    zero_init_last,
    use_attention
):
    encoder =  Encoder(
        input_channel,
        hidden_channel,
        latent_dim,
        channel_multiplier,
        num_residual_block,
        double_latent_dim, 
        resample_with_conv,
        use_conv_shortcut,
        dropout_probability,
        zero_init_last,
        use_attention
    )   
    x = torch.randn(1, 3, 16, 16)
    y = encoder(x)
    if double_latent_dim:
        assert y.shape == (1, latent_dim*2, 2, 2) 
    else:
        assert y.shape == (1, latent_dim, 2, 2) 


@pytest.mark.parametrize('output_channel', [3])
@pytest.mark.parametrize('hidden_channel', [32])
@pytest.mark.parametrize('latent_dim', [4])
@pytest.mark.parametrize('channel_multiplier', [(1, 2, 4, 8)])
@pytest.mark.parametrize('num_residual_block', [4])
@pytest.mark.parametrize('double_latent_dim', [True, False])
@pytest.mark.parametrize('resample_with_conv', [True, False])
@pytest.mark.parametrize('use_conv_shortcut', [True, False])
@pytest.mark.parametrize('dropout_probability', [0.0])
@pytest.mark.parametrize('zero_init_last', [ True, False])
@pytest.mark.parametrize('use_attention', [True, False])
def test_decoder(    
    output_channel,
    hidden_channel,
    latent_dim,
    channel_multiplier,
    num_residual_block,
    double_latent_dim, 
    resample_with_conv,
    use_conv_shortcut,
    dropout_probability,
    zero_init_last,
    use_attention
):
    decoder =  Decoder(
        output_channel,
        hidden_channel,
        latent_dim,
        channel_multiplier,
        num_residual_block,
        double_latent_dim, 
        resample_with_conv,
        use_conv_shortcut,
        dropout_probability,
        zero_init_last,
        use_attention
    )   
    
    if double_latent_dim:
        x = torch.randn(1, latent_dim*2, 2, 2)
    else:
        x = torch.randn(1, latent_dim, 2, 2)
    y = decoder(x)
    assert y.shape == (1, output_channel, 16, 16)


def test_autoencoder():
    # Get the HF autoencoder
    model_name = 'stabilityai/stable-diffusion-2-base'
    config = PretrainedConfig.get_config_dict(model_name, subfolder='vae')
    hf_autoencoder = AutoencoderKL(**config[0])
    # Make the corresponding autoencoder from this codebase
    autoencoder = AutoEncoder(input_channel=3,
                              output_channel=3,
                              hidden_channel=128,
                              latent_dim=4,
                              double_latent_dim=True,
                              channel_multiplier=(1, 2, 4, 4),
                              num_residual_block=2,
                              use_conv_shortcut=False,
                              dropout_probability=0.0,
                              resample_with_conv=True,
                              use_attention=True
                              )                            
    # Check that the number of parameters is the same
    hf_params = sum(p.numel() for p in hf_autoencoder.parameters() if p.requires_grad)
    params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    assert hf_params == params, f'{hf_params} == {params}'

