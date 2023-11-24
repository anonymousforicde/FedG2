#!/bin/bash
echo "Start running."
# <-- config1: heterogeneous NIID training
config1="train.py --partition=DirNonIID --rounds=100 --size=5 --nproc_per_gpu=4 --heter=heter"
# <-- config2: homogeneous   NIID training
config2="train.py --partition=DirNonIID --rounds=100 --size=5 --nproc_per_gpu=4 --heter=hom"

# <-- Define the config
myconfig1=${config1}" --lr=0.05 -l --model=vgg11 --dataset=CIFAR10 --alpha=0.1 --he_idx=3"
myconfig2=${config2}" --lr=0.05 -l --model=vgg11 --dataset=CIFAR10 --alpha=0.1 --he_idx=2"

# <-- run programs
echo "heter"
python ${myconfig1} --optim="fedg2"
python ${myconfig1} --optim="digfl"
python ${myconfig1} --optim="gblend"
python ${myconfig1} --optim="fedavg"
python ${myconfig1} --optim="fednova"
echo "hom"
python ${myconfig2} --optim="fedg2"
python ${myconfig2} --optim="digfl"
python ${myconfig2} --optim="gblend"
python ${myconfig2} --optim="fedavg"
python ${myconfig2} --optim="fednova"