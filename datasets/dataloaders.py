"""
Each dataset returns:
    images: torch tensor of shape [C, H, W], the image.
    labels: torch of shape [1], class of the image.
    relations: torch tensor of shape [NUM_PATCHES, NUM_PATCHES], the spatial relations of the patches.
"""


##################################################
# Imports
##################################################

import os
import subprocess
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.datasets import VOCDetection, CIFAR10, CIFAR100
import torch
import math
import torch.nn.functional as F
import tarfile

# Custom
#from datasets.encode_locations import create_connectivity_matrix, create_dist_matrix, create_angle_matrix, SSLSignal
from datasets.transforms import get_transforms, transform_enc
from datasets.tiny_imagenet import TinyImagenetDataset
from datasets.caltech import Caltech256
from datasets.flower import Flower102
from datasets.imagenet import ImageNetDataset
from datasets.oxford_pet import OxfordPet
#import utils

def get_datasets(args, transform='default', target_transform='default', patch_transform='default'):
    """
    Return the PyTorch datasets.
    """

    # Transforms
    transform = get_transforms(args)[0] if transform == 'default' else transform
    target_transform = get_transforms(args)[1] if target_transform == 'default' else target_transform
    ds_args = {
        'root': args.data_path,
        'download': True,
    }

    if args.dataset == 'tiny_imagenet':
        ds_train = TinyImagenetDataset(train=True, transform=transform['train'], 
                                       target_transform=target_transform['train'], **ds_args)
        ds_train_aug = TinyImagenetDataset(train=True, transform=transform['train_aug'], 
                                           target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = TinyImagenetDataset(train=False, transform=transform['validation'], 
                                            target_transform=target_transform['validation'], **ds_args)
        ds_test = None

    elif args.dataset in ['cifar10']:
        ds_train = CIFAR10(train=True, transform=transform['train'], target_transform=target_transform['train'], 
                           **ds_args)
        ds_train_aug = CIFAR10(train=True, transform=transform['train_aug'], 
                               target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = CIFAR10(train=False, transform=transform['validation'], 
                                target_transform=target_transform['validation'], **ds_args)
        ds_test = None

    elif args.dataset == 'cifar100':
        ds_train = CIFAR100(train=True, transform=transform['train'], target_transform=target_transform['train'], 
                            **ds_args)
        ds_train_aug = CIFAR100(train=True, transform=transform['train_aug'], 
                                target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = CIFAR100(train=False, transform=transform['validation'], 
                                 target_transform=target_transform['validation'], **ds_args)
        ds_test = None

    elif args.dataset == 'imagenet':
        ds_args = {
            'root_path': os.path.join(args.data_base_path, 'imagenet'),
        }
        ds_train = ImageNetDataset(partition='train', transform=transform['train'], target_transform=target_transform['train'], **ds_args)
        ds_train_aug = ImageNetDataset(partition='train', transform=transform['train_aug'], target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = ImageNetDataset(partition='val', transform=transform['validation'], target_transform=target_transform['validation'], **ds_args)
        ds_test = ImageNetDataset(partition='test', transform=transform['test'], target_transform=target_transform['test'], **ds_args)

    elif args.dataset == 'caltech256':
        ds_train = Caltech256(train=True, transform=transform['train'], target_transform=target_transform['train'], **ds_args)
        ds_train_aug = Caltech256(train=True, transform=transform['train_aug'], target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = Caltech256(train=False, transform=transform['validation'], target_transform=target_transform['validation'], **ds_args)
        ds_test = None

    elif args.dataset == 'flower102':
        ds_train = Flower102(split='train', transform=transform['train'], target_transform=target_transform['train'], **ds_args)
        ds_train_aug = Flower102(split='train', transform=transform['train_aug'], target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = Flower102(split='val', transform=transform['validation'], target_transform=target_transform['validation'], **ds_args)
        ds_test = Flower102(split='test', transform=transform['test'], target_transform=target_transform['test'], **ds_args)

    elif args.dataset == 'oxford_pet':
        ds_train = OxfordPet(train=True, transform=transform['train'], target_transform=target_transform['train'], **ds_args)
        ds_train_aug = OxfordPet(train=True, transform=transform['train_aug'], target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = OxfordPet(train=False, transform=transform['validation'], target_transform=target_transform['validation'], **ds_args)
        ds_test = None

    else:
        raise Exception(f'Error. Dataset {args.dataset} not supported.')

    # Datasets
    dss = {
        'train': ds_train,
        'train_aug': ds_train_aug,
        'validation': ds_validation,
        'test': None,
    }
    return dss

def get_dataloaders(args):
    """
    Return the PyTorch dataloaders.
    """

    # Datasets
    transform = 'default'
    target_transform = 'default'
    if args.backbone in ['vit', 'simsiam']:
        patch_transform = 'default'
    else:
        patch_transform = {'train': None, 'train_aug': None, 'validation': None, 'test': None}
    dss = get_datasets(args, transform=transform, target_transform=target_transform, patch_transform=patch_transform)

    # Dataloaders
    dl_args = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True,
    }
    dls = {
        'train': DataLoader(dss['train'], shuffle=False, **dl_args),
        'train_aug': DataLoader(dss['train_aug'], shuffle=True, **dl_args),
        'validation': DataLoader(dss['validation'], shuffle=False, **dl_args),
        'test':  None,
    }
    return dls
