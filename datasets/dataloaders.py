import os
import subprocess
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
import torch
import math
import torch.nn.functional as F
import tarfile
from torchvision.datasets import ImageFolder

# Custom
from datasets.transforms import get_transforms
from datasets.tiny_imagenet import TinyImagenetDataset
from datasets.caltech import Caltech256
from datasets.flower import Flower102
from datasets.imagenet import ImageNetDataset
from datasets.oxford_pet import OxfordPet


def get_datasets(args, transform='default', target_transform='default'):
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

    elif args.dataset == 'svhn':
        ds_train = SVHN(split='train', transform=transform['train'], target_transform=target_transform['train'], **ds_args)
        ds_train_aug = SVHN(split='train', transform=transform['train_aug'], target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = SVHN(split='test', transform=transform['validation'], target_transform=target_transform['validation'], **ds_args)
        ds_test = None

    else:
        raise Exception(f'Error. Dataset {args.dataset} not supported.')

    # Datasets
    dss = {
        'train': ds_train,
        'train_aug': ds_train_aug,
        'validation': ds_validation,
        'test': ds_test,
    }
    return dss
