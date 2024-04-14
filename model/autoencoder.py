import sys

from torchmetrics import MeanSquaredError
sys.path.append('model')

import torch
import torch.nn as nn 
from torch.nn import functional as F

from torchmetrics import MeanMetric, MeanSquaredError, Metric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from composer.models import ComposerModel
from composer.utils import dist
from composer.utils.file_helpers import get_file

from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from layers import ResNetBlock, Downsample, AttentionLayer,  Upsample, GradientScalingLayer
from typing import Tuple, Dict 

import lpips


class GaussianDistribution:
    def __init__(self, mean: torch.Tensor, log_var: torch.Tensor):
        self.mean = mean 
        self.log_var = log_var 
        self.var = torch.exp(log_var)
        self.stdvar = torch.exp(0.5*self.var)

    def __getitem__(self, key):
        if key == 'latent_dist':
            return self
        elif key == 'mean':
            return self.mean
        elif key == 'log_var':
            return self.log_var
        else:
            raise ValueError(key)
    
    @property
    def latent_dist(self):
        return self
    
    def sample(self):
        return self.mean + self.stdvar  * torch.rand_like(self.mean)


class AutoEncoderOutput:
    def __init__(self, x_recon):
        self.x_recon = x_recon 

    def __getitem__(self, key):
        if key == 'x_recon':
            return self.x_recon
        else:
            raise ValueError(key)

    @property
    def sample(self):
        return self.x_recon 


class Encoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        hidden_channel: int,
        latent_dim: int,
        channel_multiplier: Tuple[int, ...] = (1, 2, 4, 8),
        num_residual_block: int=4,
        double_latent_dim: bool=False, 
        resample_with_conv: bool=False,
        use_conv_shortcut: bool=False,
        dropout_probability: float=0.6,
        zero_init_last: bool=False,
        use_attention: bool=False
    ):
        super().__init__()

        self.input_channel = input_channel
        self.hidden_channel = hidden_channel 
        self.latent_dim = latent_dim 
        self.channel_multiplier = channel_multiplier
        self.num_residual_block = num_residual_block
        self.double_latent_dim = double_latent_dim 
        self.resample_with_conv = resample_with_conv
        self.use_conv_shortcut = use_conv_shortcut
        self.dropout_probability = dropout_probability
        self.zero_init_last = zero_init_last
 
        self.input_conv = nn.Conv2d(self.input_channel, self.hidden_channel, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.input_conv.weight, nonlinearity='linear')

        self.blocks = nn.ModuleList()
        block_input_channel = self.hidden_channel 
        block_output_channel = self.hidden_channel 
        for i, c in enumerate(self.channel_multiplier):
            block_output_channel = c * self.hidden_channel 
            for _ in range(self.num_residual_block):
                block = ResNetBlock(
                    input_channel=block_input_channel,
                    output_channel=block_output_channel,
                    use_conv_shortcut=self.use_conv_shortcut,
                    dropout_probability=self.dropout_probability,
                    zero_init_last=self.zero_init_last,
                )
                self.blocks.append(block)
                block_input_channel = block_output_channel
            
            if i < len(self.channel_multiplier) - 1:
                downblock = Downsample(
                    block_input_channel,
                    self.resample_with_conv
                )
                self.blocks.append(downblock)
        
        middle_block_1 = ResNetBlock(
            input_channel=block_output_channel,
            output_channel=block_output_channel,
            use_conv_shortcut=self.use_conv_shortcut,
            dropout_probability=self.dropout_probability,
            zero_init_last=self.zero_init_last
        )
        self.blocks.append(middle_block_1)

        if use_attention:
            attention = AttentionLayer(block_output_channel)
            self.blocks.append(attention)

        middle_block_2 = ResNetBlock(
            input_channel=block_output_channel,
            output_channel=block_output_channel,
            use_conv_shortcut=self.use_conv_shortcut,
            dropout_probability=self.dropout_probability,
            zero_init_last=self.zero_init_last
        )
        self.blocks.append(middle_block_2)        

        self.norm_out = nn.GroupNorm(32, num_channels=block_output_channel, eps=1e-6, affine=True)
        output_channel = 2*self.latent_dim if self.double_latent_dim else self.latent_dim 
        self.output_conv = nn.Conv2d(block_output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.output_conv.weight, nonlinearity='linear')
        self.output_conv.weight.data *= 1.6761

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_conv(x)
        for block in self.blocks:
            h = block(h)
        
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.output_conv(h)

        return h


class Decoder(nn.Module):
    def __init__(self,
                 output_channel: int = 3,
                 hidden_channel: int = 128,
                 latent_dim: int = 4,
                 channel_multiplier: Tuple[int, ...] = (1, 2, 4, 8),
                 num_residual_block: int = 4,
                 use_conv_shortcut=False,
                 dropout_probability: float = 0.0,
                 resample_with_conv: bool = True,
                 zero_init_last: bool = False,
                 use_attention: bool = True):
        super().__init__()
        self.latent_channels = latent_dim
        self.output_channels = output_channel
        self.hidden_channels = hidden_channel
        self.channel_multipliers = channel_multiplier
        self.num_residual_blocks = num_residual_block
        self.use_conv_shortcut = use_conv_shortcut
        self.dropout_probability = dropout_probability
        self.resample_with_conv = resample_with_conv
        self.zero_init_last = zero_init_last
        self.use_attention = use_attention

        # Input conv layer to get to the hidden dimensionality
        channels = self.hidden_channels * self.channel_multipliers[-1]
        self.conv_in = nn.Conv2d(self.latent_channels, channels, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv_in.weight, nonlinearity='linear')

        # Make the middle blocks
        self.blocks = nn.ModuleList()
        middle_block_1 = ResNetBlock(input_channel=channels,
                                     output_channel=channels,
                                     use_conv_shortcut=use_conv_shortcut,
                                     dropout_probability=dropout_probability,
                                     zero_init_last=zero_init_last)
        self.blocks.append(middle_block_1)

        if self.use_attention:
            attention = AttentionLayer(input_channel=channels)
            self.blocks.append(attention)

        middle_block_2 = ResNetBlock(input_channel=channels,
                                     output_channel=channels,
                                     use_conv_shortcut=use_conv_shortcut,
                                     dropout_probability=dropout_probability,
                                     zero_init_last=zero_init_last)
        self.blocks.append(middle_block_2)

        # construct the residual blocks
        block_channels = channels
        for i, cm in enumerate(self.channel_multipliers[::-1]):
            block_channels = self.hidden_channels * cm
            for _ in range(self.num_residual_blocks + 1):  # Why the +1?
                block = ResNetBlock(input_channel=channels,
                                    output_channel=block_channels,
                                    use_conv_shortcut=use_conv_shortcut,
                                    dropout_probability=dropout_probability,
                                    zero_init_last=zero_init_last)
                self.blocks.append(block)
                channels = block_channels
            # Add the upsampling block at the end, but not the very end.
            if i < len(self.channel_multipliers) - 1:
                upsample = Upsample(input_channel=block_channels, resample_with_conv=self.resample_with_conv)
                self.blocks.append(upsample)
        # Make the final layers for the output
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_channels, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_channels, self.output_channels, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv_out.weight, nonlinearity='linear')
        # Output layer is immediately after a silu. Need to account for that in init.
        # Also want the output variance to mimic images with pixel values uniformly distributed in [-1, 1].
        # These two effects essentially cancel out.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the decoder."""
        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class AutoEncoder(nn.Module):
    def __init__(
        self, 
        input_channel: int,
        output_channel: int,
        hidden_channel: int,
        latent_dim: int,
        double_latent_dim: bool = False,
        channel_multiplier: Tuple[int, ...] = (1, 2, 4, 8),
        num_residual_block: int=4,
        use_conv_shortcut=False,
        dropout_probability: float = 0.0,
        resample_with_conv: bool = False,
        zero_init_last: bool = False,
        use_attention: bool = False
    ):
        super().__init__()

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_channel = hidden_channel
        self.latent_dim = latent_dim
        self.double_latent_dim = double_latent_dim
        self.channel_multiplier = channel_multiplier
        self.num_residual_block = num_residual_block
        self.use_conv_shortcut = use_conv_shortcut
        self.dropout_probability = dropout_probability
        self.resample_with_conv = resample_with_conv
        self.zero_init_last = zero_init_last
        self.use_attention = use_attention
        self.set_extra_state(None)

        self.encoder =  Encoder(
            input_channel=self.input_channel,
            hidden_channel=self.hidden_channel,
            latent_dim=self.latent_dim,
            channel_multiplier=self.channel_multiplier,
            num_residual_block=self.num_residual_block,
            double_latent_dim=self.double_latent_dim, 
            resample_with_conv=self.resample_with_conv,
            use_conv_shortcut=self.use_conv_shortcut,
            dropout_probability=self.dropout_probability,
            zero_init_last=self.zero_init_last,
            use_attention=self.use_attention
        )  

        channel = 2*self.latent_dim if self.double_latent_dim else self.letent_dim 
        self.quant_conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.quant_conv.weight, nonlinearity='linear')
        # KL divergence is minimized when mean is 0.0 and log variance is 0.0
        # However, this corresponds to no information in the latent space.
        # So, init these such that latents are mean 0 and variance 1, with a rough snr of 1
        self.quant_conv.weight.data[:channel // 2] *= 0.707
        self.quant_conv.weight.data[channel // 2:] *= 0.707
        if self.quant_conv.bias is not None:
            self.quant_conv.bias.data[channel // 2:].fill_(-0.9431)

        self.post_quant_conv = nn.Conv2d(self.latent_dim, self.latent_dim, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.post_quant_conv.weight, nonlinearity='linear')
        self.decoder =  Decoder(
            output_channel=self.output_channel,
            hidden_channel=self.hidden_channel,
            latent_dim=self.latent_dim,
            channel_multiplier=self.channel_multiplier,
            num_residual_block=self.num_residual_block,
            resample_with_conv=self.resample_with_conv,
            use_conv_shortcut=self.use_conv_shortcut,
            dropout_probability=self.dropout_probability,
            zero_init_last=self.zero_init_last,
            use_attention=self.use_attention
        )   

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_extra_state(self):
        return #{'config': self.config}

    def set_extra_state(self, state):
        pass

    def get_last_layer_weight(self) -> torch.Tensor:
        return self.decoder.conv_out.weight

    def encode(self, h) -> GaussianDistribution:
        h = self.encoder(h)
        moments = self.quanv_conv(h)
        mean, log_var = moments[:, :self.latent_dim], moments[:, self.latent_dim:]
        return GaussianDistribution(mean, log_var)

    def decode(self, h) -> AutoEncoderOutput:
        z = self.post_quant_conv(h)
        x_recon = self.decoder(z)
        return AutoEncoderOutput(x_recon)

    def forward(self, x):
        encoded_dist = self.encoder(x)
        z = encoded_dist.sample()
        x_recon = self.decoder(z)['x_recon']
        return {'x_recon': x_recon, 'latents': z, 'mean': encoded_dist.mean, 'log_var': encoded_dist.log_var}


class NlayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator.

    Based on code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

    Args:
        input_channels (int): Number of input channels. Default: `3`.
        num_filters (int): Number of filters in the first layer. Default: `64`.
        num_layers (int): Number of layers in the discriminator. Default: `3`.
    """

    def __init__(self, input_channels: int = 3, num_filters: int = 64, num_layers: int = 3):
        super().__init__()

        self.input_channels = input_channels 
        self.num_filters = num_filters
        self.num_layers = num_layers 

        # Input layer
        self.block = nn.Sequential()
        input_conv = nn.Conv2d(self.input_channels, self.num_filters, kernel_size=4, stride=2, padding=1)
        nn.init.kaiming_normal_(input_conv.weight, nonlinearity='linear')
        nonlinearity = nn.LeakyReLU(0.2, True)
        self.block.extend([input_conv, nonlinearity])
        
        # Hidden layer
        num_filters = self.num_filters
        out_filters = self.num_filters
        for n in range(1, self.num_layers):
            out_filter = self.num_filters * min(2**n, 8)
            hidden_conv  = nn.Conv2d(num_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False)
            num_filters = out_filters
            nn.init.kaiming_normal_(hidden_conv.weight, nonlinearity='linear')
            norm = nn.BatchNorm2d(out_filters)
            nonlinearity = nn.LeakyReLU(0.2, True)
            self.block.extend([hidden_conv, norm, nonlinearity])

        # Output layer
        out_filters = self.num_filters * min(2**self.num_filters, 8)
        out_conv = nn.Conv2d(num_filters, out_filters, kernel_size=4, stride=1, padding=1, bias=False)
        num_filters = out_filters
        nn.init.kaiming_normal_(out_conv.weight, nonlinearity='linear')
        norm = nn.BatchNorm2d(out_filters)
        nonlinearity = nn.LeakyReLU(0.2, True)
        self.block.extend([out_conv, norm, nonlinearity])

        out_layer = nn.Conv2d(num_filters, 1, kernel_size=4, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(out_layer.weight, nonlinearity='linear')
        out_layer.weight.data *= 0.1
        self.block.append(out_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AutoEncoderLoss(nn.Module):
    """Loss function for training an autoencoder. Includes LPIPs and a discriminator.

    Args:
        input_key (str): Key for the input to the model. Default: `image`.
        ae_output_channels (int): Number of output channels. Default: `3`.
        learn_log_var (bool): Whether to learn the output log variance. Default: `True`.
        log_var_init (float): Initial value for the log variance. Default: `0.0`.
        kl_divergence_weight (float): Weight for the KL divergence loss. Default: `1.0`.
        lpips_weight (float): Weight for the LPIPs loss. Default: `0.25`.
        discriminator_weight (float): Weight for the discriminator loss. Default: `0.5`.
        discriminator_num_filters (int): Number of filters in the first layer of the discriminator. Default: `64`.
        discriminator_num_layers (int): Number of layers in the discriminator. Default: `3`.
    """

    def __init__(self,
                 input_key: str = 'image',
                 ae_output_channels: int = 3,
                 learn_log_var: bool = True,
                 log_var_init: float = 0.0,
                 kl_divergence_weight: float = 1.0,
                 lpips_weight: float = 0.25,
                 discriminator_weight: float = 0.5,
                 discriminator_num_filters: int = 64,
                 discriminator_num_layers: int = 3):
        super().__init__()
        self.input_key = input_key
        self.ae_output_channels = ae_output_channels
        self.learn_log_var = learn_log_var
        self.log_var_init = log_var_init
        self.kl_divergence_weight = kl_divergence_weight
        self.lpips_weight = lpips_weight
        self.discriminator_weight = discriminator_weight
        self.discriminator_num_filters = discriminator_num_filters
        self.discriminator_num_layers = discriminator_num_layers

        if self.learn_log_var:
            self.log_var = nn.Parameter(torch.zeros(size=()))
        else:
            self.log_var = torch.zeros(size=())
        self.log_var.data.fill_(self.log_var_init)

        # Lpips loss
        self.lpips = lpips.LPIPS(net='vgg').eval()
        for param in self.lpips.parameters():
            param.requires_grad_(False)
        for param in self.lpips.net.parameters():
            param.requires_grad_(False)

        # Discriminator 
        self.discriminator = NlayerDiscriminator(
            input_channels=self.ae_output_channels,
            num_filters=self.discriminator_num_filters, 
            num_layers=self.discriminator_num_layers
        )
        self.scale_gradient = GradientScalingLayer()
        self.scale_gradient.register_full_backward_hook(self.scale_gradient.backward_hook)

    def set_discriminator_weight(self, weight):
        self.discriminator_weight = weight
    
    def calc_discriminator_adaptive_weight(self, nll_loss, fake_loss, last_layer):
        # Need to ensure the grad scale from the discriminator back to 1.0 to get the right norm
        self.scale_gradient.set_scale(1.0)
        # Get the grad norm from the nll loss
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        # Get the grad norm for the discriminator loss
        disc_grads = torch.autograd.grad(fake_loss, last_layer, retain_graph=True)[0]

        # Calculate the updated discriminator weight based on the grad norms
        nll_grads_norm = torch.norm(nll_grads)
        disc_grads_norm = torch.norm(disc_grads)
        disc_weight = nll_grads_norm / (disc_grads_norm + 1e-4)
        disc_weight = torch.clamp(disc_weight, 0.0, 1e4).detach()
        disc_weight *= self.discriminator_weight
        # Set the discriminator weight. It should be negative to reverse gradients into the autoencoder.
        self.scale_gradient.set_scale(-disc_weight.item())
        return disc_weight, nll_grads_norm, disc_grads_norm

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], last_layer: torch.Tensor):
        losses = {}

        # l1 loss
        ae_loss = F.l1_loss(outputs['x_recon'], batch[self.input_key], reduction='none')
        num_output_elements = ae_loss[0].numel()
        losses['ae_loss'] = ae_loss 

        # Lpips loss
        recon_image = outputs['x_recon'].clamp(-1, 1)
        target_image = batch[self.input_key].clamp(-1, 1)
        lpips_loss = self.lpips(recon_image, target_image)

        # nll loss 
        rec_loss = ae_loss + self.lpips_weight * lpips_loss 
        nll_loss = rec_loss / torch.exp(self.log_var) + self.log_var + 2
        nll_loss = nll_loss.mean()
        losses['nll_loss'] = nll_loss
        losses['output_variance'] = torch.exp(self.log_var)

        # Discriminator loss
        real = self.discriminator(batch[self.input_key])
        fake = self.discriminator(self.scale_gradient(outputs['x_recon']))
        real_loss = F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
        fake_loss = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
        losses['disc_real_loss'] = real_loss
        losses['disc_fake_loss'] = fake_loss
        losses['disc_loss'] = 0.5 * (real_loss + fake_loss)

        # Update the adaptive discriminator loss
        disc_weight, disc_nll_norm, disc_grads_norm = self.calc_discriminator_adaptive_weight(
            nll_loss, fake_loss, last_layer
        )
        losses['disc_weight'] = disc_weight 
        losses['disc_nll_norm'] = disc_nll_norm
        losses['disc_fake_norm'] = disc_grads_norm

        # KL divergence loss
        mean = outputs['mean']
        log_var = outputs['log_var']
        kl_div_loss = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
        num_latent_elements = mean[0].numel()
        losses['kl_div_loss'] = kl_div_loss.mean()

        # Combine loss
        dimensionality_weight = num_latent_elements / num_output_elements
        total_loss = losses['nll_loss'] + self.kl_divergence_weight * dimensionality_weight * losses['kl_div_loss']
        total_loss += losses['disc_loss']
        losses['total'] = total_loss
        return losses


class ComposerAutoEncoder(ComposerModel):
    def __init__(self, model: AutoEncoder, autoencoder_loss: AutoEncoderLoss, input_key: str='image'):
        super().__init__()
        self.model = model
        self.autoencoder_loss = autoencoder_loss 
        self.input_key = input_key

        # Train metrics
        train_metrics = [MeanSquaredError()]
        self.train_metrics = {metric.__class__.__name__: metric for metric in train_metrics}

        # Set up val metrics
        psnr_metric = PeakSignalNoiseRatio(data_range=2.0)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0)
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        val_metrics = [MeanSquaredError(), MeanMetric(), lpips_metric, psnr_metric, ssim_metric]
        self.val_metrics = {metric.__class__.__name__: metric for metric in val_metrics}

    def get_last_layer_weight(self) -> torch.Tensor:
        return self.model.get_last_layer_weight()

    def forward(self, batch):
        return self.model(batch[self.input_key])

    def loss(self, output, batch):
        last_layer = self.model.get_last_layer_weight()
        return self.autoencoder_loss(output, batch, last_layer)

    def eval_forward(self, batch, outputs=None):
        if outputs is not None:
            return outputs

        outputs = self.forward(batch)
        return outputs

    def get_metric(self, is_train: bool=False):
        if is_train:
            metrics = self.train_metrics 
        else:
            metrics = self.eval_metrics

        if isinstance(metrics, Metric):
            metrics_dict = {metrics.__layer__.__name__: metrics}
        elif isinstance(metrics, list):
            metrics_dict = {metric.__layer__.__name__: metric for metric in metrics}
        else:
            metrics_dict = {}
            for name, metric in metrics.items():
                assert isinstance(metric, Metric)
                metrics_dict[name] = metric

        return metrics_dict

    def update_metric(self, batch, outputs, metric):
        clamped_imgs = outputs['x_recon'].clamp(-1, 1)
        if isinstance(metric, MeanMetric):
            metric.update(torch.square(outputs['latents']))
        elif isinstance(metric, LearnedPerceptualImagePatchSimilarity):
            metric.update(clamped_imgs, batch[self.input_key])
        elif isinstance(metric, PeakSignalNoiseRatio):
            metric.update(clamped_imgs, batch[self.input_key])
        elif isinstance(metric, StructuralSimilarityIndexMeasure):
            metric.update(clamped_imgs, batch[self.input_key])
        elif isinstance(metric, MeanSquaredError):
            metric.update(outputs['x_recon'], batch[self.input_key])
        else:
            metric.update(outputs['x_recon'], batch[self.input_key])

    
class ComposerDiffusersAutoEncoder(ComposerModel):
    """Composer wrapper for the Huggingface Diffusers Autoencoder.

    Args:
        model (diffusers.AutoencoderKL): Diffusers autoencoder to train.
        autoencoder_loss (AutoEncoderLoss): Auto encoder loss module.
        input_key (str): Key for the input to the model. Default: `image`.
    """

    def __init__(self, model: AutoencoderKL, autoencoder_loss: AutoEncoderLoss, input_key: str = 'image'):
        super().__init__()
        self.model = model
        self.autoencoder_loss = autoencoder_loss
        self.input_key = input_key

        # Set up train metrics
        train_metrics = [MeanSquaredError()]
        self.train_metrics = {metric.__class__.__name__: metric for metric in train_metrics}
        # Set up val metrics
        psnr_metric = PeakSignalNoiseRatio(data_range=2.0)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0)
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        val_metrics = [MeanSquaredError(), MeanMetric(), lpips_metric, psnr_metric, ssim_metric]
        self.val_metrics = {metric.__class__.__name__: metric for metric in val_metrics}

    def get_last_layer_weight(self) -> torch.Tensor:
        """Get the weight of the last layer of the decoder."""
        return self.model.decoder.conv_out.weight

    def forward(self, batch):
        encoder_output = self.model.encode(batch[self.input_key], return_dict=True)
        assert isinstance(encoder_output, AutoencoderKLOutput)
        latent_dist = encoder_output['latent_dist']
        latents = latent_dist.sample()
        mean, log_var = latent_dist.mean, latent_dist.logvar
        output_dist = self.model.decode(latents, return_dict=True)
        assert isinstance(output_dist, DecoderOutput)
        recon = output_dist.sample
        return {'x_recon': recon, 'latents': latents, 'mean': mean, 'log_var': log_var}

    def loss(self, outputs, batch):
        last_layer = self.get_last_layer_weight()
        return self.autoencoder_loss(outputs, batch, last_layer)

    def eval_forward(self, batch, outputs=None):
        if outputs is not None:
            return outputs
        outputs = self.forward(batch)
        return outputs

    def get_metrics(self, is_train: bool = False):
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics

        if isinstance(metrics, Metric):
            metrics_dict = {metrics.__class__.__name__: metrics}
        elif isinstance(metrics, list):
            metrics_dict = {metrics.__class__.__name__: metric for metric in metrics}
        else:
            metrics_dict = {}
            for name, metric in metrics.items():
                assert isinstance(metric, Metric)
                metrics_dict[name] = metric

        return metrics_dict

    def update_metric(self, batch, outputs, metric):
        clamped_imgs = outputs['x_recon'].clamp(-1, 1)
        if isinstance(metric, MeanMetric):
            metric.update(torch.square(outputs['latents']))
        elif isinstance(metric, LearnedPerceptualImagePatchSimilarity):
            metric.update(clamped_imgs, batch[self.input_key])
        elif isinstance(metric, PeakSignalNoiseRatio):
            metric.update(clamped_imgs, batch[self.input_key])
        elif isinstance(metric, StructuralSimilarityIndexMeasure):
            metric.update(clamped_imgs, batch[self.input_key])
        elif isinstance(metric, MeanSquaredError):
            metric.update(outputs['x_recon'], batch[self.input_key])
        else:
            metric.update(outputs['x_recon'], batch[self.input_key])