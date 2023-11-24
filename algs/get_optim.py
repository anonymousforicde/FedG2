from .FedG2   import FedG2
from .DIGFL   import DIGFL
from .GBlend  import GBlend
from .FedNova import FedNova
from .FedProx import FedProx

opts = {
    'fedavg'       : FedProx, # mu = 0        
    'fedg2'        : FedG2,
    'digfl'        : DIGFL,
    'gblend'       : GBlend,
    'fedprox'      : FedProx,
    'fednova'      : FedNova,}

def get_optim(args, model, ratio):
    if args.optim == "fedg2":
        optimizer = FedG2(model.parameters(), 
                          ratio=ratio, 
                          gmf=0, 
                          mu=args.mu,
                          rho=args.rho,
                          lr=args.lr,
                          momentum=args.momentum, 
                          weight_decay=1e-4, 
                          nesterov=False)
    elif args.optim == "digfl":
        optimizer = DIGFL(model.parameters(), 
                          ratio=ratio, 
                          gmf=0, 
                          mu=args.mu,
                          lr=args.lr,
                          momentum=args.momentum, 
                          weight_decay=1e-4, 
                          nesterov=False)        
    elif args.optim == "gblend":
        optimizer = GBlend(model.parameters(), 
                          ratio=ratio, 
                          gmf=0, 
                          mu=args.mu,
                          lr=args.lr,
                          momentum=args.momentum, 
                          weight_decay=1e-4, 
                          nesterov=False)            
    elif args.optim == "fedavg":
        optimizer = FedProx(model.parameters(), 
                            ratio=ratio, 
                            gmf=0, 
                            mu=0,
                            lr=args.lr,
                            momentum=args.momentum, 
                            weight_decay=1e-4, 
                            nesterov=False)
    elif args.optim == "fedprox":
        if args.mu==0:
            print('warning: mu is 0 for fedprox')
        optimizer = FedProx(model.parameters(), 
                            ratio=ratio, 
                            gmf=0, 
                            mu=args.mu,
                            lr=args.lr,
                            momentum=args.momentum, 
                            weight_decay=1e-4, 
                            nesterov=False)        
    elif args.optim == "fednova":
        optimizer = FedNova(model.parameters(), 
                            ratio=ratio, 
                            gmf=0, 
                            mu=args.mu,
                            lr=args.lr,
                            momentum=args.momentum, 
                            weight_decay=1e-4, 
                            nesterov=False)              
    else:
        raise ValueError("Invalid optim: {}".format(args.optim))
    
    return optimizer