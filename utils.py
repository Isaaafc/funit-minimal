import numpy as np
import matplotlib.pyplot as plt

# Reference: https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
