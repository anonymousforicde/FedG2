import os
import datetime
import argparse

import torch
import torch.distributed as dist
import torch.nn as nn
from   torch.multiprocessing import Process

import copy
import random
import numpy   as np

from model import get_model
from algs  import get_optim
from data  import get_dataset
from utils import train_sgd,compute_accuracy,get_global_grad
   
##############
#  Training  #
##############

def training(rank, size, args):
    # <-- seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    gpu = int(rank//args.nproc_per_gpu)
    torch.cuda.set_device(gpu)
    world_size = dist.get_world_size()
    
    # <-- Load Dataset 
    datasize_list   = args.data_prop
    train_set, test_set, val_set, bsz, ratio_list = get_dataset(args.bsz, datasize_list, args)
    dataratio_list = [x for x in ratio_list]    
    
    # <-- Load Model
    np.random.seed(args.seed)
    if rank==0 and args.model=='alex':
        # get drop ratio for AlexNet in Helios
        drop_ratio_list = np.random.randint(low=1, high=5, size=world_size)/10
        print(drop_ratio_list)
        drop_ratio = drop_ratio_list[dist.get_rank()]   
    else:
        drop_ratio = 0.0
    model = get_model(args.model, args.dataset, ratio=drop_ratio).cuda(gpu)
    global_model = get_model(args.model, args.dataset).cuda(gpu)
    if args.pre_model:
        load_model = torch.load('checkpoint/model_last.pkl')
        model.load_state_dict(load_model['dict'])
        model_rnd = load_model['epoch']+1
        print("start from round:" + str(model_rnd))
    else:
        model_rnd = 0

    # <-- Load optim, criterion (loss func) and lr_scheduler  
    optimizer = get_optim(args, model, dataratio_list[dist.get_rank()])
    criterion   = nn.CrossEntropyLoss().cuda(gpu)
    lr_scheduler= get_scheduler(optimizer, args.rounds)

    trainer = train_sgd 
         
    # <-- training            
    for rnd in range(model_rnd, args.rounds):
        rnd += 1
        global_model = copy.deepcopy(model)
        global_optim, val_loss = get_global_grad(global_model, val_set, args.lr)
                    
        # <-- local training
        if args.heter == 'random':
            np.random.seed(args.seed+rnd)
            args.cmp_prop = np.random.randint(low=1, high=args.he_idx, size=world_size)
            if rank==0:
                print(args.cmp_prop)
        local_epochs = args.cmp_prop[dist.get_rank()]
        
        train_loss = 0.0
        for t in range(local_epochs):
            loss = trainer(gpu, model, criterion, optimizer, train_set)
            train_loss += loss

        # <-- change lr in unit of rounds
        if args.scheduler:
            lr_scheduler.step()
        
        # <-- sync model & global_loss
        dist.barrier()
        optimizer.average(global_optim, local_epochs, len(val_set), val_loss, train_loss)
        dist.barrier()      

        # <-- record global results
        if rank==0:
            top1_acc, test_loss = compute_accuracy(model, test_set, False)
            print('round: ', rnd, ' top1_acc: ', top1_acc)   
        dist.barrier()   

    dist.barrier()
    if rank == dist.get_rank():
        print('rank ' + str(dist.get_rank()) + ' training over!')




def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
    parser.add_argument('--seed',      default=1234, type=int, help='seed')
    parser.add_argument('--size',      default=4, type=int, help='total gpu num')
    parser.add_argument('--nproc_per_gpu', default=1, type=int, help='# processes per gpu')
    parser.add_argument('--rounds',    default=30, type=int, help='communicate rounds')
    parser.add_argument('--model',     default='resnet_18', help='model name')
    parser.add_argument('--pre_model', default=False, help='Pre-trained model')
    parser.add_argument('--dataset',   default='CIFAR10', help='dataset name')
    parser.add_argument('--partition', default='IID', help='data partition method')
    parser.add_argument('--alpha',     default=0.2, type=float, 
                                       help='control the non-iidness of dataset')
    parser.add_argument('--mu',        default=0.0, type=float, help='mu for fedprox')
    parser.add_argument('--rho',       default=1.0, type=float, help='rho for FedG2')    
    parser.add_argument('--bsz',       default=32, type=int, help='batch_size per rank')
    parser.add_argument('--optim',     default='fedavg', help='optimizer')
    parser.add_argument('--momentum',  default=0.0, type=float, help='momentum')
    parser.add_argument('--lr',        default=0.01, type=float, help='learning rate')
    parser.add_argument('--scheduler', '-l', action='store_true', 
                                       help='whether to use the lr_scheduler')
    parser.add_argument('--cmp_prop',  default=[1,1,1,1], help='local epochs')
    parser.add_argument('--heter',     default='hom', help='cmp heter mode')
    parser.add_argument('--data_prop', default=[1,1,1,1], help='data size list')
    parser.add_argument('--he_idx',    default=8, type=int, help='controls heterogeneity')         
    args = parser.parse_args()
    return args

def init_process(rank, size, fn, args, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend,rank=rank,world_size=size,timeout=datetime.timedelta(0,7200))
    fn(rank, size, args)

##############
#   Helper   #
##############

def get_scheduler(optimizer, rounds):
    milestones1 = 0.5*rounds
    milestones2 = 0.75*rounds
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestones1,milestones2], gamma=0.1)
    return scheduler



if __name__=="__main__":
    args = get_args()
    world_size = args.size * args.nproc_per_gpu
     
    heter_list  = np.ones(world_size, dtype=int)
    for i in range(int(world_size/2)):
        heter_list[i]=args.he_idx
    np.random.seed(args.seed)
    rand_idx = 2 if args.he_idx==1 else args.he_idx
    list_opts = {
        'hom'   : np.ones(world_size, dtype=int)*args.he_idx,
        'heter' : heter_list,
        'random': np.random.randint(low=1, high=rand_idx, size=world_size),}
    args.cmp_prop  = list_opts[args.heter]

    print(args.optim)

    data_num = np.ones(world_size, dtype=int)
    args.data_prop = [ x/np.sum(data_num) for x in data_num ]
        
    processes = []
    for rank in range(world_size):
        p = Process(target=init_process, 
                    args=(rank, world_size, training, args))
        p.start()
        processes.append(p)
          
    for p in processes:
        p.join()
