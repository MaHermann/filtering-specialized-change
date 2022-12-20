import random

import torchvision.transforms.functional as TF


class NormalizeImg:
    """Normalize only the image tensor, not the mask."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        x, mask = sample
        x = TF.normalize(x, self.mean, self.std)
        return x, mask

class RandomFlip:
    """Randomly flip the images in a sample."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        x, mask = sample
        if random.random() < self.p:
            x =  TF.hflip(x)
            mask =  TF.hflip(mask.unsqueeze(0))
        return x, mask.squeeze(0)


class RandomRot:
    """Randomly rotate the images in a sample."""

    def __call__(self, sample):
        x, mask = sample
        angle = random.choice([0, 90, 180, 270])
        x = TF.rotate(x, angle)
        mask = TF.rotate(mask.unsqueeze(0), angle)
        return x, mask.squeeze(0)
