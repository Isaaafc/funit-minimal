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
from trainer import FUNIT_Trainer

num_epochs = 1
image_size = 32
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

def get_data_loaders(image_size, train=True):
    ds = datasets.cifar_data(image_size=image_size, train=train)

    content_weights = [1 if label < splits[0] and label >= splits[1] else 0 for _, label in ds]
    class_weights = [1 if label < splits[1] and label >= splits[2] else 0 for _, label in ds]

    content_sampler = WeightedRandomSampler(weights=content_weights, num_samples=len([x for x in content_weights if x == 1]))
    class_sampler = WeightedRandomSampler(weights=class_weights, num_samples=len([x for x in class_weights if x == 1]))

    content_loader = DataLoader(ds, sampler=content_sampler)
    class_loader = DataLoader(ds, sampler=class_sampler)

    return content_loader, class_loader

def recon_criterion(self, input, target):
    return torch.mean(torch.abs(input - target))

if __name__ == '__main__':
    content_loader, class_loader = get_data_loaders(image_size, True)
    
    hyperparameters = {
        'source_classes': 1,
        'lr': 0.001,
        'beta1': 0.9,
        'beta2': 0.999,
        'weight_decay': 0.001
    }
    
    trainer = FUNIT_Trainer(hyperparameters)

    # Training
    for n in range(num_epochs):
        for i, (content_image, class_image) in enumerate(zip(content_loader, class_loader)):
            content_var = content_image[0]
            class_var = class_image[0].unsqueeze(0)

            trainer.dis_update(content_var, class_var)