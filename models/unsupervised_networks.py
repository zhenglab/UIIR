import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.optim import lr_scheduler
from torchvision import models
from util.tools import *
from scipy.linalg import block_diag
from util import util
from . import base_networks as networks_init


###############################################################################
# Helper Functions
###############################################################################

def define_G(netG='ris',init_type='normal', init_gain=0.02, opt=None):
    if netG == 'ri':
        net = RIGenerator(opt)
    elif netG == 'ris':
        net = RISGenerator(opt)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    net = networks_init.init_weights(net, init_type, init_gain)
    net = networks_init.build_model(opt, net)
    return net

def define_D(netD='patchgan',init_type='normal', init_gain=0.02, opt=None):
    if netD == 'patchgan':
        net = networks_init.NLayerDiscriminator(opt.input_nc, n_layers=opt.n_layers_D, norm_layer='nn.InstanceNorm2d')
    elif netD == 'sim':
        net = SimDiscriminator(opt)
    else:
        raise NotImplementedError('D model name [%s] is not recognized' % netD)
    net = networks_init.init_weights(net, init_type, init_gain)
    net = networks_init.build_model(opt, net)
    return net

def define_F(input_nc, netF, use_dropout=False, init_type='normal', init_gain=0.02, opt=None):
    if netF == 'global_pool':
        net = PoolingF()
    elif netF == 'reshape':
        net = ReshapeF()
    elif netF == 'sample':
        net = PatchSampleF(use_mlp=False, init_type=init_type, init_gain=init_gain, nc=opt.netF_nc, opt=opt)
    elif netF == 'mlp_sample':
        net = PatchSampleF(use_mlp=True, init_type=init_type, init_gain=init_gain, nc=opt.netF_nc, opt=opt)
    elif netF == 'strided_conv':
        net = StridedConvF(init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF)
    net = networks_init.init_weights(net, init_type, init_gain)
    net = networks_init.build_model(opt, net)
    return net

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

##############################################################################
# Classes
##############################################################################
class RIGenerator(nn.Module):
    def __init__(self, opt=None):
        super(RIGenerator, self).__init__()
        self.reflectance_dim = 256
        self.device = opt.device
        r_enc_n_res = 4
        r_dec_n_res = 0
        i_enc_n_res = 4
        i_dec_n_res = 0
        self.reflectance_enc = ContentEncoderForConstrast(opt.n_downsample, r_enc_n_res, opt.input_nc, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=opt.pad_type)
        self.reflectance_dec = ContentDecoder(opt.n_downsample, r_dec_n_res, self.reflectance_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=opt.pad_type)

        self.illumination_enc = ContentEncoder(opt.n_downsample, i_enc_n_res, opt.input_nc, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=opt.pad_type)
        self.illumination_dec = ContentDecoder(opt.n_downsample, i_dec_n_res, self.illumination_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=opt.pad_type)

    def forward(self, inputs, layers=[], encode_only=False):
        if encode_only:
            feats = self.reflectance_enc(inputs, layers, encode_only)
            return feats
        else:
            r_content = self.reflectance_enc(inputs)
            i_content = self.illumination_enc(inputs)

            reflectance = self.reflectance_dec(r_content)
            reflectance = reflectance / 2 +0.5

            illumination = self.illumination_dec(i_content)
            illumination = illumination / 2 + 0.5
            
            reconstruction = reflectance*illumination

            return reconstruction, reflectance, illumination

class RISGenerator(nn.Module):
    def __init__(self, opt=None):
        super(RISGenerator, self).__init__()
        self.reflectance_dim = 256
        self.device = opt.device
        r_enc_n_res = 4
        r_dec_n_res = 0
        i_enc_n_res = 4
        i_dec_n_res = 0
        self.reflectance_enc = ContentEncoderForConstrast(opt.n_downsample, r_enc_n_res, opt.input_nc, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=opt.pad_type)
        self.reflectance_dec = ContentDecoder(opt.n_downsample, r_dec_n_res, self.reflectance_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=opt.pad_type)

        self.scattering_enc = ContentEncoder(opt.n_downsample, r_enc_n_res, opt.input_nc, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=opt.pad_type)
        self.scattering_dec = ContentDecoder(opt.n_downsample, r_dec_n_res, self.reflectance_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=opt.pad_type)

        self.illumination_enc = ContentEncoder(opt.n_downsample, i_enc_n_res, opt.input_nc, self.reflectance_dim, opt.ngf, 'in', opt.activ, pad_type=opt.pad_type)
        self.illumination_dec = ContentDecoder(opt.n_downsample, i_dec_n_res, self.illumination_enc.output_dim, opt.output_nc, opt.ngf, 'ln', opt.activ, pad_type=opt.pad_type)

    def forward(self, inputs, layers=[], encode_only=False):
        if encode_only:
            feats = self.reflectance_enc(inputs, layers, encode_only)
            return feats
        else:
            r_content = self.reflectance_enc(inputs)
            i_content = self.illumination_enc(inputs)

            reflectance = self.reflectance_dec(r_content)
            reflectance = reflectance / 2 +0.5

            illumination = self.illumination_dec(i_content)
            illumination = illumination / 2 + 0.5

            scattering = self.scattering_dec(self.scattering_enc(inputs))
            scattering = scattering/2+0.5
            reconstruction = reflectance*illumination+scattering

            return reconstruction, reflectance, illumination, scattering


##################################################################################
# Encoder and Decoders
##################################################################################

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, output_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
           
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        if not dim == output_dim:
            self.model += [Conv2dBlock(dim, output_dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class ContentEncoderForConstrast(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, output_dim, dim, norm, activ, pad_type):
        super(ContentEncoderForConstrast, self).__init__()
        self.model = []
        self.model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_dim, dim, kernel_size=7, padding=0, bias=False),
                 nn.InstanceNorm2d(dim),
                 nn.LeakyReLU(0.2, True)]
        # downsampling blocks
        for i in range(n_downsample):

            self.model += [nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
                          nn.InstanceNorm2d(dim * 2),
                          nn.LeakyReLU(0.2, True),
                        #   Downsample(dim * 2)
                          ]
            dim *= 2
           
        # residual blocks
        # self.model += [ResBlocks(n_res, dim, norm='ln', activation=activ, pad_type=pad_type)]
        for i in range(n_res):
            self.model += [ResBlock(dim, norm=norm, activation=activ, pad_type=pad_type)]
        if not dim == output_dim:
            self.model += [Conv2dBlock(dim, output_dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, input, layers=[], encode_only=False):
        # return self.model(x)
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake

class ContentDecoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, output_dim, dim, norm, activ, pad_type):
        super(ContentDecoder, self).__init__()
        self.model = []
        dim = input_dim
        # residual blocks
        # self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]

        # upsampling blocks
        for i in range(n_downsample):
            self.model += [
                nn.Upsample(scale_factor=2),
                Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)
            ]
            dim //= 2

        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

     

##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, input_dim//2, norm=norm, activation=activ)]
        dim = input_dim//2
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim//2, norm=norm, activation=activ)]
            dim = dim//2
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', groupcount=16):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        self.norm_type = norm
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'remove_render':
            self.norm = RemoveRender(norm_dim)
        elif norm == 'grp':
            self.norm = nn.GroupNorm(groupcount, norm_dim)
        
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ConvTranspose2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', groupcount=16):
        super(ConvTranspose2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'grp':
            self.norm = nn.GroupNorm(groupcount, norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=padding, bias=self.use_bias))
        else:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=padding, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, opt=None):
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.opt = opt
        self.mlp_0 = nn.Sequential(*[nn.Linear(3, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
        self.mlp_1 = nn.Sequential(*[nn.Linear(128, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
        self.mlp_2 = nn.Sequential(*[nn.Linear(256, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
        self.mlp_3 = nn.Sequential(*[nn.Linear(256, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
        self.mlp_4 = nn.Sequential(*[nn.Linear(256, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
        self.mlp_init = True
    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            if mlp_id > 0:
                input_nc = feat.shape[1]
                mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.LayerNorm(self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
                mlp.cuda()
                setattr(self, 'mlp_%d' % mlp_id, mlp)
        net = networks_init.init_weights(self, self.init_type, self.init_gain)
        networks_init.build_model(self.opt, net)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  #(b,h*w,c)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))] 
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1) 
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)

                x_sample = mlp(x_sample) 
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids   

class SimDiscriminator(nn.Module):

    def __init__(self, opt):
       
        super(SimDiscriminator, self).__init__()
        sequence = [Conv2dBlock(opt.input_nc, opt.ndf, 7, 1, 3, norm='ln', activation=opt.activ, pad_type=opt.pad_type)]
        kw = 4
        padw = 1
        nf_mult = 1
        nf_mult_prev = 1
        dim = opt.ndf
        for n in range(0, opt.n_layers_D):  # gradually increase the number of filters
            sequence += [
                Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm='ln', activation=opt.activ, pad_type=opt.pad_type),
            ]
            dim *= 2
        sequence += [Conv2dBlock(dim, dim, 3, 1, 1, norm='none', activation='none', pad_type=opt.pad_type)]
        sequence += [
            nn.AvgPool2d(32)
        ]
        self.model = nn.Sequential(*sequence)
        self.linear = nn.Sequential(
            nn.Linear(dim, dim*2, bias=True),
            LayerNorm(dim*2),
            nn.LeakyReLU(0.2, inplace=True), # hidden layer
            nn.Linear(dim*2, dim*2)
        )
        
    def forward(self, inputs):
        output = self.model(inputs)
        output = self.linear(output.flatten(1))
        return output
