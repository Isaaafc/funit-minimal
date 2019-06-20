import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/munit/models.py
class ResidualBlock(nn.Module):
    def __init__(self, features, norm="in"):
        super(ResidualBlock, self).__init__()

        self.weight = None
        self.bias = None
        
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
        self.residual1 = nn.Sequential(
            ResidualBlock(features=512),
            nn.ReLU(True)
        )
        self.residual2 = nn.Sequential(
            ResidualBlock(features=512),
            nn.ReLU(True)
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

        self.residual1 = ResidualBlock(features=512, norm='adain')
        self.residual2 = ResidualBlock(features=512, norm='adain')

    # cls_out: decoded class latent code
    def forward(self, x, cls_out):
        cls_out = cls_out.squeeze()

        self.residual1.bias = cls_out[:, :512]
        self.residual1.weight = cls_out[:, 512:2 * 512]

        cls_out = cls_out[:2 * 512]

        self.residual2.bias = cls_out[:, :512]
        self.residual2.weight = cls_out[:, 512:2 * 512]

        print(self.residual1.bias, self.residual1.weight)
        print(self.residual2.bias, self.residual2.weight)

        x = self.residual1(x)
        x = self.residual2(x)

        return x

# Class code input: FC-256 > FC-256 > FC-256
# Content code input: AdaInResBlk-512 > AdaInResBlk-512 > ConvTrans-256 > ConvTrans-128 > ConvTrans-64 > ConvTrans-3
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.adain = AdaInResBlock()
        self.fc1 = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, self.adain.total_num_params),
            nn.ReLU(True)
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=3, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(3),
            nn.ReLU(True)
        )
    
    def forward(self, x, cls_in):
        cls_out = self.fc1(cls_in)
        cls_out = self.fc2(cls_out)
        cls_out = self.fc3(cls_out)

        print(cls_out.size())

        x = self.adain(x, cls_out)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.contentencoder = ContentEncoder()
        self.classencoder = ClassEncoder()
        self.decoder = Decoder()

    def forward(self, x, classes):
        class_codes = list()

        for y in classes:
            class_codes.append(self.classencoder(y))

        content_code = self.contentencoder(x)
        
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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x