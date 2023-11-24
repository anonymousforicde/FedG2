from random import Random
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist

class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

""" get data subsets with different length and Non-IID data """

class Heter_DataPartitioner(object):
    def __init__(self, data, dataname, dataratio_list, seed=1234, partition='IID', alpha=0):
        self.data = data
        
        if   partition == 'DirNonIID':
            self.partitions, self.ratio = self.__getDirichletData__(data, dataname, 
                                               dataratio_list, seed, alpha, 0)
        elif partition == 'IID':
            self.partitions = []
            self.ratio      = dataratio_list
            data_len        = len(data)
            indexes         = [x for x in range(0, data_len)]
            rng             = Random() 
            rng.seed(seed)
            rng.shuffle(indexes)

            for frac in dataratio_list:
                part_len = int(data_len * frac)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

    def __getDirichletData__(self, data, dataname, psizes, seed, alpha, num):
        n_nets = len(psizes)
        if dataname == 'CIFAR100':
            K = 100
        else:
            K = 10
        if dataname == 'SVHN':
            labelList = np.array(data.labels)
        else:
            labelList = np.array(data.targets)   
        min_size = 0
        N = len(labelList)
        np.random.seed(2020)

        net_dataidx_map = {}
        while min_size < K:
            idx_batch = [[] for _ in range(n_nets)]
            # <-- for each class in the dataset
            for k in range(K):
                idx_k = np.where(labelList == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                # <-- Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            
        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp
        if dist.get_rank()==0:
            print('Data statistics: %s' % str(net_cls_counts))

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i])) 
        local_sizes = np.array(local_sizes)
        weights = local_sizes/np.sum(local_sizes)
        #print(weights)

        return idx_batch, weights