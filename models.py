import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/munit/models.py
# class ResidualBlock(nn.Module):
#     def __init__(self, features, norm="in"):
#         super(ResidualBlock, self).__init__()

#         norm_layer = AdaptiveInstanceNorm2d if norm == "adain" else nn.InstanceNorm2d

#         self.block = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(features, features, 3),
#             norm_layer(features),
#             nn.ReLU(inplace=True),
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(features, features, 3),
#             norm_layer(features),
#         )

#     def assign_adain_params(self, adain_params):
#         # assign the adain_params to the AdaIN layers in model
#         for m in self.block.children():
#             if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
#                 mean = adain_params[:, :m.num_features]
#                 std = adain_params[:, m.num_features:2*m.num_features]
#                 m.bias = mean.contiguous().view(-1)
#                 m.weight = std.contiguous().view(-1)
#                 if adain_params.size(1) > 2*m.num_features:
#                     adain_params = adain_params[:, 2*m.num_features:]

#     def get_num_adain_params(self):
#         # return the number of AdaIN parameters needed by the model
#         num_adain_params = 0
#         for m in self,block.children():
#             if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
#                 num_adain_params += 2*m.num_features
#         return num_adain_params

#     def forward(self, x):
#         return x + self.block(x)

# class AdaptiveInstanceNorm2d(nn.Module):
#     """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

#     def __init__(self, num_features, eps=1e-5, momentum=0.1):
#         super(AdaptiveInstanceNorm2d, self).__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         # weight and bias are dynamically assigned
#         self.weight = None
#         self.bias = None
#         # just dummy buffers, not used
#         self.register_buffer("running_mean", torch.zeros(num_features))
#         self.register_buffer("running_var", torch.ones(num_features))

#     def forward(self, x):
#         assert (
#             self.weight is not None and self.bias is not None
#         ), "Please assign weight and bias before calling AdaIN!"
#         b, c, h, w = x.size()
#         running_mean = self.running_mean.repeat(b)
#         running_var = self.running_var.repeat(b)

#         # Apply instance norm
#         x_reshaped = x.contiguous().view(1, b * c, h, w)

#         out = nn.functional.batch_norm(
#             x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps
#         )

#         return out.view(b, c, h, w)

#     def __repr__(self):
#         return self.__class__.__name__ + "(" + str(self.num_features) + ")"

# class LayerNorm(nn.Module):
#     def __init__(self, num_features, eps=1e-5, affine=True):
#         super(LayerNorm, self).__init__()
#         self.num_features = num_features
#         self.affine = affine
#         self.eps = eps

#         if self.affine:
#             self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
#             self.beta = nn.Parameter(torch.zeros(num_features))

#     def forward(self, x):
#         shape = [-1] + [1] * (x.dim() - 1)
#         mean = x.view(x.size(0), -1).mean(1).view(*shape)
#         std = x.view(x.size(0), -1).std(1).view(*shape)
#         x = (x - mean) / (std + self.eps)

#         if self.affine:
#             shape = [1, -1] + [1] * (x.dim() - 2)
#             x = x * self.gamma.view(*shape) + self.beta.view(*shape)
#         return x

# Conv-64 > Conv-128 > Conv-256 > Conv-512 > Conv-1024 > AvgPool12x2
class ClassEncoder(nn.Module):
    def __init__(self):
        super(ClassEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.ReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.ReLU(True)
        )
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)

        return x

# Conv-64 > Conv-128 > Conv-256 > Conv-512 > ResBlk-512 > ResBlk-512
# Each followed by instance normalization and ReLU
class ContentEncoder(nn.Module):
    def __init__(self):
        super(ContentEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(512),
            nn.ReLU(True)
        )
        self.residual = ResBlocks(2, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.residual(x)

        return x

# class AdaInResBlock(nn.Module):
#     total_num_params = 4 * 512

#     def __init__(self):
#         super(AdaInResBlock, self).__init__()

#         self.residual = nn.Sequential(
#             ResidualBlock(features=512, norm='adain'),
#             ResidualBlock(features=512, norm='adain')
#         )

#     # cls_out: decoded class latent code
#     def forward(self, x, cls_out):
#         for m in self.residual.children():
#             mean = cls_out[:, :512]
#             std = cls_out[:, 512:2 * 512]

#             m.bias = mean.contiguous().view(-1)
#             m.weight = std.contiguous().view(-1)

#             cls_out = cls_out[:2 * 512]

#         x = self.residual(x)

#         return x

# # Class code input: FC-256 > FC-256 > FC-256
# # Content code input: AdaInResBlk-512 > AdaInResBlk-512 > ConvTrans-256 > ConvTrans-128 > ConvTrans-64 > ConvTrans-3
# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()

#         self.residual1 = ResidualBlock(features=512, norm='adain')
#         self.residual2 = ResidualBlock(features=512, norm='adain')
#         self.fc1 = nn.Sequential(
#             nn.Linear(1, 256),
#             nn.ReLU(True)
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(256, 256),
#             nn.ReLU(True)
#         )
#         self.fc3 = nn.Sequential(
#             nn.Linear(256, self.residual1.get_num_adain_params()),
#             nn.ReLU(True)
#         )
#         self.conv1 = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels=512, out_channels=256, kernel_size=4, 
#                 stride=2, padding=1, bias=False
#             ),
#             nn.InstanceNorm2d(256),
#             nn.ReLU(True)
#         )
#         self.conv2 = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels=256, out_channels=128, kernel_size=4, 
#                 stride=2, padding=1, bias=False
#             ),
#             nn.InstanceNorm2d(128),
#             nn.ReLU(True)
#         )
#         self.conv3 = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels=128, out_channels=64, kernel_size=4, 
#                 stride=2, padding=1, bias=False
#             ),
#             nn.InstanceNorm2d(64),
#             nn.ReLU(True)
#         )
#         self.conv4 = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels=64, out_channels=3, kernel_size=4, 
#                 stride=2, padding=1, bias=False
#             ),
#             nn.InstanceNorm2d(3),
#             nn.ReLU(True)
#         )

#     def forward(self, x, cls_in):
#         cls_out = self.fc1(cls_in)
#         cls_out = self.fc2(cls_out)
#         cls_out = self.fc3(cls_out)

#         print(cls_out.size())

#         x = self.residual1(x, cls_out)
        
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)

#         return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.contentencoder = ContentEncoder()
        self.classencoder = ClassEncoder()
        self.decoder = Decoder(n_upsample=4, n_res=2, dim=512, output_dim=3)

    def forward(self, x, classes):
        class_codes = list()

        for y in classes:
            class_codes.append(self.classencoder(y))

        content_code = self.contentencoder(x)
        
        class_codes = torch.stack(class_codes, dim=0)
        class_code = torch.mean(class_codes, dim=0)

        self.assign_adain_params(class_code, self.decoder)

        out = self.decoder(content_code)

        return out

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

class DiscriminatorLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscriminatorLayer, self).__init__()

        self.layer = nn.Sequential(
            ConvResBlock(in_channels, out_channels),
            ResBlock(out_channels),
            nn.AvgPool2d(2, 2)
        )
    
    def forward(self, x):
        x = self.layer(x)

        return x
       
class Discriminator(nn.Module):
    # num_classes: No. of source classes
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64,
            kernel_size=3, stride=1, padding=1
        )
        self.layer1 = DiscriminatorLayer(64, 128)
        self.layer2 = DiscriminatorLayer(128, 256)
        self.layer3 = DiscriminatorLayer(256, 512)
        self.layer4 = DiscriminatorLayer(512, 1024)
        self.residual = ResBlocks(2, 1024)
        self.conv2 = nn.Conv2d(
            in_channels=1024, out_channels=num_classes,
            kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.residual(x)
        x = self.conv2(x)

        return x

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

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
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
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

class ConvResBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvResBlock, self).__init__()
        self.actv1 = nn.LeakyReLU(True)
        self.conv1 = nn.Conv2d(input_dim, output_dim, 3, 1, 1)
        self.actv2 = nn.LeakyReLU(True)
        self.conv2 = nn.Conv2d(output_dim, output_dim, 3, 1, 1)
        if input_dim != output_dim:
            self.convs = nn.Conv2d(input_dim, output_dim, 1, 1)

    def forward(self, x):
        x = self.actv1(x)
        s = self.convs(x) if hasattr(self, "convs") else x
        x = self.conv1(x)
        x = self.conv2(self.actv2(x))
        return x + s

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
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