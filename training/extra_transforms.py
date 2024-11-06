import torch
import math
import random
from PIL import Image, ImageOps, ImageFilter
import numpy as np

class RandomRectRotation(object):
    """Randomly rotates the given PIL.Image to 0,90,180,270 degrees. Each angle is equally probable
    """

    def __call__(self, img):
        angs=[0,90,180,270]
        ang=random.choice(angs)
        if ang!=0:
            return img.rotate(ang,expand=True)
        return img

class Unnormalize(object):
    """Unnormalize an image normalized with normalize"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class GaussianBlur(object):
    """
    In 50% of cases will blur the image with a Gaussian kernel.
    
    """
    def __init__(self, radmax=0.8):
        self.radmax = radmax
    
    def __call__(self, img):
        if random.uniform(0, 1) > 0.5:
            radius = random.uniform(0.1, self.radmax)
            img = img.filter(ImageFilter.GaussianBlur(radius))
        return img
