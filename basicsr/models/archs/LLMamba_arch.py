import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from .SS2D_arch import TMSS2D, ITSS2D, CSSM
from .IFA_arch import IFA
from .block import LSA, LCA
import sys
import torchvision.transforms as transforms
import cv2 as cv
import numpy as np
import numbers
print(sys.path)
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)  # noqa: E741
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")

def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):

    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride
    )

def shift_back(inputs, step=2):

    [bs, nC, row, col] = inputs.shape

    down_sample = 256 // row

    step = float(step) / float(down_sample * down_sample)

    out_col = row

    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]

    return inputs[:, :, :, :out_col]

class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class Dilated_Resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dilated_Resblock, self).__init__()

        sequence = list()
        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=3, dilation=(3, 3)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1))

        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x) + x
        return out

class GRM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GRM, self).__init__()

        self.conv_head = Depth_conv(in_channels, out_channels)
        self.dilated_block_HL = Dilated_Resblock(out_channels, out_channels)
        self.conv_tail = Depth_conv(out_channels, in_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        x = self.conv_head(x)
        x = self.dilated_block_HL(x)
        out = self.conv_tail(x)
        return out + residual


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
        )

        self.net1 = nn.Conv2d(dim * mult, dim, 1, 1, bias=False)

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        out = self.net1(out)
        return out.permute(0, 2, 3, 1)

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class SpatialLayer(nn.Module):

    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2,d_state = 16):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                LSA(dim, nn.Conv2d),
                LayerNorm(dim, LayerNorm_type='WithBias'),
                TMSS2D(d_model=dim,dropout=0,d_state =d_state),
                ITSS2D(d_model=dim,dropout=0,d_state =d_state),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, brightness_weights, gradient_weights):
        for (lsa, norm, tmss2d, itss2d, ffn) in self.blocks:
            z = x
            z1 = lsa(x)
            x = norm(x)
            y = x.permute(0, 2, 3, 1)
            x = tmss2d(y, brightness_weights, gradient_weights) + z1.permute(0, 2, 3, 1)
            x = itss2d(x, brightness_weights, gradient_weights) + x
            x = x + z.permute(0, 2, 3, 1)
            x = ffn(x) + x
            x = x.permute(0, 3, 1, 2)
        return x

class ChannelLayer(nn.Module):

    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2, d_state=16):

        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                LCA(dim),
                LayerNorm(dim, LayerNorm_type='WithBias'),
                CSSM(d_model=dim, dropout=0, d_state=d_state),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        for (lca, norm, cssm, ffn) in self.blocks:
            z = x
            z1 = lca(x)
            x = norm(x)
            y = x.permute(0, 2, 3, 1)
            x = cssm(y) + z1.permute(0, 2, 3, 1)
            x = x + z.permute(0, 2, 3, 1)
            x = ffn(x) + x
            x = x.permute(0, 3, 1, 2)
        return x

class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4],d_state= 16):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
       
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                SpatialLayer(dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim, d_state=d_state),
                ChannelLayer(dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim, d_state=d_state),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(3, 3, 4, 2, 1, bias=False),
                nn.Conv2d(3, 3, 4, 2, 1, bias=False),
            ]))
            dim_level *= 2
            d_state *= 2

        self.bottleneck = SpatialLayer(dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1], d_state=d_state)
        self.bottleneck1 = ChannelLayer(dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1], d_state=d_state)

        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.ConvTranspose2d(3, 3, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.ConvTranspose2d(3, 3, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                SpatialLayer(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim, heads=(dim_level // 2) // dim, d_state=d_state),
                ChannelLayer(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                     heads=(dim_level // 2) // dim, d_state=d_state)
            ]))
            dim_level //= 2
            d_state //= 2

        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def compute_brightness_weights(self, x):
        x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
        brightness = 0.2126 * x[:, 0, :, :] + 0.7152 * x[:, 1, :, :] + 0.0722 * x[:, 2, :, :]
        brightness_weights = brightness.view(x.shape[0], -1)
        return brightness_weights

    def histogram_equalization(self, img):
        img_list = [self._equalize_single_image(img[i]) for i in range(img.shape[0])]
        img_equalized = torch.stack(img_list)

        return img_equalized

    def _equalize_single_image(self, img):
        img = img.permute(1, 2, 0).cpu().detach().numpy()
        img = (img * 255.0).astype(np.uint8)
        hsv_image = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        v = hsv_image[:, :, 2]
        equalized_v = cv.equalizeHist(v)
        hsv_image[:, :, 2] = equalized_v
        enhanced_image = cv.cvtColor(hsv_image, cv.COLOR_HSV2RGB)
        img_equalized = torch.tensor(enhanced_image).float() / 255.0
        img_equalized = img_equalized.permute(2, 0, 1)  # Remove the batch dimension
        return img_equalized

    def compute_gradient_map(self, img):
        img = img.to('cuda' if torch.cuda.is_available() else 'cpu')
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_x = sobel_x.to(img.device).repeat(img.shape[1], 1, 1, 1)
        sobel_y = sobel_y.to(img.device).repeat(img.shape[1], 1, 1, 1)
        gradient_x = F.conv2d(img, sobel_x, padding=1, groups=img.shape[1])
        gradient_y = F.conv2d(img, sobel_y, padding=1, groups=img.shape[1])
        gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
        return gradient_magnitude

    def compute_gradient_weights(self, img):
        gradient_weights = img.mean(dim=1)
        gradient_weights = gradient_weights.view(img.shape[0], -1)
        return gradient_weights

    def forward(self, x, y):
        z = x
        z1 = y

        fea = self.embedding(x)

        count = 0
        fea_encoder = []
        for (spatial, channel, FeaDownSample, IlluFeaDownsample, GraDownsample) in self.encoder_layers:
            count+=1
            brightness_weights = self.compute_brightness_weights(z)
            gradient_weights = self.compute_gradient_weights(z1)
            fea = spatial(fea, brightness_weights, gradient_weights)
            fea = channel(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            z = IlluFeaDownsample(z)
            z1 = GraDownsample(z1)

        brightness_weights = self.compute_brightness_weights(z)
        gradient_weights = self.compute_gradient_weights(z1)
        fea = self.bottleneck(fea, brightness_weights, gradient_weights)
        fea = self.bottleneck1(fea)
        
        count = 0

        for i, (FeaUpSample, FeaUpSample2, FeaUpSample3, Fusion, spatial, channel) in enumerate(self.decoder_layers):
            count+=1
            fea = FeaUpSample(fea)
            z = FeaUpSample2(z)
            z1 = FeaUpSample3(z1)
            brightness_weights = self.compute_brightness_weights(z)
            gradient_weights = self.compute_gradient_weights(z1)
            fea = Fusion(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            fea = spatial(fea, brightness_weights, gradient_weights)
            fea = channel(fea)

        out = self.mapping(fea) + x
        return out

def histogram_equalization1(img):
    img_list = [_equalize_single_image1(img[i]) for i in range(img.shape[0])]
    img_equalized = torch.stack(img_list)
    return img_equalized

def _equalize_single_image1(img):
    img = img.permute(1, 2, 0).cpu().detach().numpy()
    img = (img * 255.0).astype(np.uint8)
    hsv_image = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    v = hsv_image[:, :, 2]
    equalized_v = cv.equalizeHist(v)
    hsv_image[:, :, 2] = equalized_v
    enhanced_image = cv.cvtColor(hsv_image, cv.COLOR_HSV2RGB)
    img_equalized = torch.tensor(enhanced_image).float() / 255.0
    img_equalized = img_equalized.permute(2, 0, 1)  # Remove the batch dimension
    return img_equalized

def compute_gradient_map1(img):
    img = img.to('cuda' if torch.cuda.is_available() else 'cpu')
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_x = sobel_x.to(img.device).repeat(img.shape[1], 1, 1, 1)
    sobel_y = sobel_y.to(img.device).repeat(img.shape[1], 1, 1, 1)
    gradient_x = F.conv2d(img, sobel_x, padding=1, groups=img.shape[1])
    gradient_y = F.conv2d(img, sobel_y, padding=1, groups=img.shape[1])
    gradient_magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
    return gradient_magnitude

class LLMamba_Single_Stage(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1],d_state = 16):
        super(LLMamba_Single_Stage, self).__init__()
        self.gradient = GRM(3,64)
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level, num_blocks=num_blocks, d_state=d_state)
    
    def forward(self, img):
        x = compute_gradient_map1(img)
        x = self.gradient(x)
        output_img = self.denoiser(img, x)
        return output_img, x


class LLMamba(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1, 1, 1], d_state=16):
        super(LLMamba, self).__init__()
        self.stage = stage
        modules_body = [LLMamba_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2, num_blocks=num_blocks, d_state=d_state)
                        for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)
    
    def forward(self, x):
        out, gra = self.body(x)
        return [gra, out]


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    model = LLMamba(stage=1,n_feat=40,num_blocks=[1,2,2]).cuda()
    print(model)
    inputs = torch.randn((1, 3, 256, 256)).cuda()
    model(inputs)
    flops = FlopCountAnalysis(model,inputs)
    n_param = sum([p.nelement() for p in model.parameters()])
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')