from random import Random
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from .partitioner import *


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std  = (0.2023, 0.1994, 0.2010)


def heter2_partition_CIFAR10(bsz, dataratio_list, args):
    dataname = 'CIFAR10'
    size = dist.get_world_size()
    if args.model=='alex':
        transform_train = transforms.Compose([
            transforms.Resize([32,32]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])    
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    train_data = datasets.CIFAR10(root='../data', train=True, 
                                  download=False, transform=transform_train)

    partition = Heter_DataPartitioner(train_data, dataname, dataratio_list,
                                       partition=args.partition, alpha=args.alpha)
    
    ratio = partition.ratio
    partition = partition.use(dist.get_rank())
    #print(partition.__len__())
    train_dataloader = DataLoader(partition,batch_size=bsz,shuffle=True,pin_memory=True)
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_data = datasets.CIFAR10(root='../data', train=False, 
                                 download=False, transform=transform_test)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=size)
    val_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=size)    
    
    return train_dataloader, test_dataloader, val_dataloader, bsz, ratio

cifar100_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
cifar100_std  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

def heter2_partition_CIFAR100(bsz, dataratio_list, args):
    dataname = 'CIFAR100'
    size = dist.get_world_size()
    train_data = datasets.CIFAR100('../data', train=True, download=False,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(32, padding=4),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(15), # <-- data augmentation
                                       transforms.ToTensor(),
                                       transforms.Normalize(cifar100_mean,cifar100_std),])
                                  )
    partition = Heter_DataPartitioner(train_data, dataname, dataratio_list,
                                       partition=args.partition, alpha=args.alpha)
    ratio = partition.ratio
    partition = partition.use(dist.get_rank())
    #print(partition.__len__())
    train_dataloader = DataLoader(partition,batch_size=bsz,shuffle=True,drop_last=True)

    test_data = datasets.CIFAR100('../data', train=False, download=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(cifar100_mean,cifar100_std),])
                                 )
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=size)

    val_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=size)    
    
    return train_dataloader, test_dataloader, val_dataloader, bsz, ratio
    

def heter2_partition_SVHN(bsz, dataratio_list, args):
    dataname = 'SVHN'
    size = dist.get_world_size()
    train_data = datasets.SVHN('../data', split='train', download=False,
                               transform=transforms.Compose([
                                   transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(cifar10_mean,cifar10_std),])
                               )
    partition = Heter_DataPartitioner(train_data, dataname, dataratio_list,
                                       partition=args.partition, alpha=args.alpha)
    ratio = partition.ratio
    partition = partition.use(dist.get_rank())
    #print(partition.__len__())
    train_dataloader = DataLoader(partition,batch_size=bsz,shuffle=True,drop_last=True)

    test_data = datasets.SVHN('../data', split='test', download=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(cifar10_mean,cifar10_std),])
                              )
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=size)
    val_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=size)    
    
    return train_dataloader, test_dataloader, val_dataloader, bsz, ratio


