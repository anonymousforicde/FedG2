from .dataset  import heter2_partition_CIFAR10, heter2_partition_CIFAR100, heter2_partition_SVHN

dataset_options = [
    'CIFAR10'
    'CIFAR100'
    'SVHN'    ]

def get_dataset(bsz, datasize_list, args):
    if args.dataset == "CIFAR10":
        train_set, test_set, val_set, bsz, ratio_list = heter2_partition_CIFAR10(bsz, datasize_list, args)
    elif args.dataset == "CIFAR100":
        train_set, test_set, val_set, bsz, ratio_list = heter2_partition_CIFAR100(bsz, datasize_list, args)
    elif args.dataset == "SVHN":
        train_set, test_set, val_set, bsz, ratio_list = heter2_partition_SVHN(bsz, datasize_list, args)
    else:
        raise ValueError("Invalid dataset: {}".format(args.dataset))
    
    return train_set, test_set, val_set, bsz, ratio_list