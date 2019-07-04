import models
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd.variable import Variable

class FUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(FUNIT_Trainer, self).__init__()
        self.generator = models.Generator()
        self.discriminator = models.Discriminator(hyperparameters['source_classes'])

        self.gan_loss = nn.BCELoss()
        # The paper was not clear about this loss, as it references VAE papers using BSE but uses L1 itself
        self.content_reconstruction_loss = nn.L1Loss(size_average=True, reduce=True)
        # Same as content reconstruction loss: unclear
        self.feature_matching_loss = nn.L1Loss(size_average=True, reduce=True)

        lr = hyperparameters['lr'] 

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.discriminator.parameters())
        gen_params = list(self.generator.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

    def forward(self, content_image, class_image):
        self.eval()
        fake_data = self.generator(content_image, class_image)
        self.train()
    
    def dis_update(self, content_image, class_images):
        self.dis_opt.zero_grad()
        
        fake_data = self.generator(content_image, class_images)
        x = self.discriminator(fake_data)
        loss = 0

        loss_recon = self.content_reconstruction_loss(fake_data, content_image)
        loss_fm = self.feature_matching_loss()
        loss_gan = self.gan_loss()
        
        self.dis_opt.step()
        