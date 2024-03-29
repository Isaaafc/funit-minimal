from torchvision import transforms, datasets
import os

DATA_DIR = './data'

def cifar_data(image_size=32, out_dir=DATA_DIR, train=True):
    compose = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ]
    )
    out_dir = os.path.join(out_dir, 'cifar')
    
    return datasets.CIFAR10(root=out_dir, train=train, transform=compose, download=True)