from .resnet import *
from .alexnet import AlexNet
from .vgg import *
from .Net import *

NUM_CLASSES = {
    "MNIST": 10,
    "CIFAR10": 10,
    "SVHN": 10,
    "CIFAR100": 100,
    "tinyimagenet": 200,
}

def get_model(model_name, dataset, ratio=0.01):
    if dataset in ('MNIST','CIFAR10','CIFAR100','SVHN','tinyimagenet'):
        num_classes = NUM_CLASSES[dataset]

    if model_name=='res18':
        net=resnet18(num_classes=num_classes)    

    if model_name=='alex':
        net=AlexNet(num_classes=num_classes, ratio=ratio)   

    elif model_name=='vgg11':
        net=vgg11(num_classes=num_classes)

    elif model_name=='vgg19':
        net=vgg19(num_classes=num_classes)
        
    elif model_name=='Net':
        net=Net(num_classes=num_classes)      
    
    return net