# motivation https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py

import argparse
from ast import arg, parse
from errno import EISDIR
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('images', exist_ok=True)

# parser = argparse.ArgumentParser()

img_shape = (chaneels, img_size, img_size)

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
        