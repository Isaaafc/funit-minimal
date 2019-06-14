import models
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.variable import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
import datasets

gan_loss = nn.BCELoss()
# The paper was not clear about this loss, as it references VAE papers using BSE but uses L1 itself
content_reconstruction_loss = nn.L1Loss()
# Same as content reconstruction loss: unclear
feature_matching_loss = nn.L1Loss()

data_loader = datasets.cifar_data()
num_epochs = 1
image_size = 28

generator = models.Generator(image_size * image_size)

def images_to_vectors(images, image_size):
    return images.view(images.size(0), image_size * image_size)

def vectors_to_images(vectors, image_size):
    return vectors.view(vectors.size(0), 1, image_size, image_size)

def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n

for n in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data_loader):
        real_data = Variable(images_to_vectors(real_batch, image_size))

        fake_data = generator(noise(real_data.size(0))).detach()