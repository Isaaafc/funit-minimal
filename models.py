import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/munit/models.py
class ResidualBlock(nn.Module):
    def __init__(self, features, norm="in"):
        super(ResidualBlock, self).__init__()

        norm_layer = AdaptiveInstanceNorm2d if norm == "adain" else nn.InstanceNorm2d

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
        )

    def forward(self, x):
        return x + self.block(x)

class AdaptiveInstanceNorm2d(nn.Module):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, h, w)

        out = nn.functional.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps
        )

        return out.view(b, c, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"

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
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

# Conv-64 > Conv-128 > Conv-256 > Conv-512 > Conv-1024 > AvgPool12x2
class ClassEncoder(nn.Module):
    def __init__(self):
        super(ClassEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=3, out_channels=64, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.ReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.ReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.ReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.ReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.ReLU(0.2, inplace=True)
        )
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)

        return x

# Conv-64 > Conv-128 > Conv-256 > Conv-512 > ResBlk-512 > ResBlk-512
# Each followed by instance normalization and ReLU
class ContentEncoder(nn.Module):
    def __init__(self):
        super(ContentEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=3, out_channels=64, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(128),
            nn.ReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(256),
            nn.ReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(512),
            nn.ReLU(0.2, inplace=True)
        )
        self.residual1 = nn.Sequential(
            ResidualBlock(features=512),
            nn.ReLU(0.2, inplace=True)
        )
        self.residual2 = nn.Sequential(
            ResidualBlock(features=512),
            nn.ReLU(0.2, inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.residual1(x)
        x = self.residual2(x)

        return x

class AdaInResBlock(nn.Module):
    total_num_params = 4 * 512

    def __init__(self):
        super(AdaInResBlock, self).__init__()

        self.residual = nn.Sequential(
            ResidualBlock(features=512, norm='adain'),
            ResidualBlock(features=512, norm='adain')
        )

    # cls_out: decoded class latent code
    def forward(self, x, cls_out):
        for m in self.residual.children():
            mean = cls_out[:, :512]
            std = cls_out[:, 512:2 * 512]

            m.bias = mean.contiguous.view(-1)
            m.weight = std.contiguous.view(-1)

            cls_out = cls_out[:2 * 512]
        
        x = self.residual(x)

        return x

# Class code input: FC-256 > FC-256 > FC-256
# Content code input: AdaIN-ResBlk-512 > AdaInResBlk-512 > ConvTrans-256 > ConvTrans-128 > ConvTrans-64 > ConvTrans-3
class Decoder(nn.Module):
    def __init__(self, in_dim):
        super(Decoder, self).__init__()

        self.adain = AdaInResBlock()
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(0.2, inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(0.2, inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, self.adain.total_num_params),
            nn.ReLU(0.2, inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(256),
            nn.ReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(128),
            nn.ReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=3, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(3),
            nn.ReLU(0.2, inplace=True)
        )
    
    def forward(self, x, cls_in):
        cls_out = self.fc1(cls_in)
        cls_out = self.fc2(cls_out)
        cls_out = self.fc3(cls_out)

        x = self.adain(x, cls_out)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x

class Generator(nn.Module):
    def __init__(self, in_dim):
        super(Generator, self).__init__()

        self.contentencoder = ContentEncoder()
        self.classencoder = ClassEncoder()
        self.decoder = Decoder(in_dim)

    def forward(self, x, classes):
        content_code = self.contentencoder(x)
        class_codes = list()

        for y in classes:
            class_codes.append(self.classencoder(y))
        
        class_codes = torch.stack(class_codes, dim=0)
        class_code = torch.mean(class_codes, dim=0)

        out = self.decoder(content_code, class_code)

        return out

class DiscriminatorLayer(nn.Module):
    def __init__(self, features):
        super(DiscriminatorLayer, self).__init__()

        self.layer = nn.Sequential(
            ResidualBlock(features=features),
            ResidualBlock(features=features),
            nn.AvgPool2d(2, 2)
        )
    
    def forward(self, x):
        x = self.layer(x)

        return x
        
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=64, out_channels=128,
            kernel_size=3, stride=1, padding=1
        )
        self.layer1 = DiscriminatorLayer(128)
        self.layer2 = DiscriminatorLayer(256)
        self.layer3 = DiscriminatorLayer(512)
        self.layer4 = DiscriminatorLayer(1024)
        self.residual = nn.Sequential(
            ResidualBlock(features=1024),
            ResidualBlock(features=2014)
        )
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

        return x