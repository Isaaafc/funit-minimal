import models
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.variable import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
import datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import  WeightedRandomSampler

gan_loss = nn.BCELoss()
# The paper was not clear about this loss, as it references VAE papers using BSE but uses L1 itself
content_reconstruction_loss = nn.L1Loss()
# Same as content reconstruction loss: unclear
feature_matching_loss = nn.L1Loss()

num_epochs = 1
image_size = 28
num_workers = 4
# Num of: source class, target class, test class
splits = [7, 2, 1]

def images_to_vectors(images, image_size):
    return images.view(images.size(0), image_size * image_size)

def vectors_to_images(vectors, image_size):
    return vectors.view(vectors.size(0), 1, image_size, image_size)

def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n

generator = models.Generator(image_size * image_size)
ds = datasets.cifar_data(64)

content_weights = [1 if label < splits[0] and label >= splits[1] else 0 for data, label in ds]
class_weights = [1 if label < splits[1] and label >= splits[2] else 0 for data, label in ds]

content_sampler = WeightedRandomSampler(weights=content_weights, num_samples=len([x for x in content_weights if x == 1]))
class_sampler = WeightedRandomSampler(weights=class_weights, num_samples=len([x for x in class_weights if x == 1]))

content_loader = DataLoader(ds, sampler=content_sampler)
class_loader = DataLoader(ds, sampler=class_sampler)

print(len(content_loader), len(class_loader))

# Training
for n in range(num_epochs):
    pass