##################################################
# Imports
##################################################

from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np
from PIL import ImageFilter
import random

# Custom
from datasets import constants


def transform_enc(labels):
    """
    Args:
        labels: tensor of shape [9]
    
    Output:
        labels_matrix: tensor of shape [num_classes, 3, 3]
    """
    enc = compute_label_encoding() # [8, 9, 9]
    enc = torch.tensor(enc)
    enc = enc[:, labels, :][:, :, labels]
    return enc


def smart_resize(x, h, w):
    """
    x: PIL Image.
    """
    h_in, w_in = x.size
    resize = False
    h_out = min(h_in, h)
    w_out = min(w_in, w)
    if h_out < h:
        resize = True
    if w_out < w:
        resize = True
    if resize:
        x = transforms.Resize((h, w))(x)
    return x

def gray2rgb_ifneeded(x):
    return x.repeat(3, 1, 1) if x.shape[0] == 1 else x

def rgba2rgb_ifneeded(x):
    return x[:3] if x.shape[0] > 3 else x

def get_transforms(args):
    """
    Return the transformations for the datasets.
    """
    if args.dataset in ['imagenet', 'caltech256', 'flower102', 'oxford_pet']:
        trans = {
            'train_aug': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                gray2rgb_ifneeded,
                rgba2rgb_ifneeded,
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
            'train': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                gray2rgb_ifneeded,
                rgba2rgb_ifneeded,
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
            'validation': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                gray2rgb_ifneeded,
                rgba2rgb_ifneeded,
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
            'test': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                gray2rgb_ifneeded,
                rgba2rgb_ifneeded,
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
        }
        trans_target = {
            'train': None,
            'train_aug': None,   
            'validation': None,   
            'test': None,   
        }

    elif args.dataset in ['cifar10', 'cifar100', 'tiny_imagenet']:
        trans = {
            'train_aug': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                gray2rgb_ifneeded,
                rgba2rgb_ifneeded,
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                gray2rgb_ifneeded if args.dataset == 'tiny_imagenet' else lambda x: x,
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
            'validation': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                gray2rgb_ifneeded if args.dataset == 'tiny_imagenet' else lambda x: x,
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                gray2rgb_ifneeded if args.dataset == 'tiny_imagenet' else lambda x: x,
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
        }
        trans_target = {
            'train': None,
            'train_aug': None,   
            'validation': None,   
            'test': None,   
            'train_simsiam': None,
        }

    else:
        print('No transformations match. No transformations are applied to the datasets.')
        trans = {
            'train': None,
            'train_aug': None,
            'validation': None,
            'test': None,
            'train_simsiam': None,
        }

        trans_target = {
            'train': None,
            'train_aug': None,   
            'validation': None,   
            'test': None,   
            'train_simsiam': None,
        }
        patch_trans = {
            'train': None,
            'train_aug': None,   
            'validation': None,   
            'test': None,   
            'train_simsiam': None,
        }

    return trans, trans_target
