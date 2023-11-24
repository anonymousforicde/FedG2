import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required
from .comm_helpers import communicate, flatten_tensors, unflatten_tensors
import threading
import numpy as np


class GBlend(Optimizer):
    def __init__(self, params, ratio, gmf, mu=0, lr=required, momentum=0, 
                 dampening=0, weight_decay=0, nesterov=False, variance=0):
        
        self.gmf      = gmf
        self.ratio    = ratio
        self.momentum = momentum
        self.mu       = mu
        self.rounds   = 1
        self.weight   = 0.0
        self.overfit  = 0.0
        self.local_steps = 0

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, variance=variance)                         
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(GBlend, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(GBlend, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)                 

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(p.data,alpha= weight_decay)
                
                param_state = self.state[p]
                if 'old_init' not in param_state:
                    param_state['old_init'] = torch.clone(p.data).detach()

                local_lr = group['lr']                    

                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha= 1 - dampening)
                    if nesterov:
                        d_p.add_(buf,alpha=momentum)
                    else:
                        d_p = buf

                # update accumalated local updates
                if 'cum_grad' not in param_state:
                    param_state['cum_grad'] = torch.clone(d_p).detach()
                    param_state['cum_grad'].mul_(local_lr)

                else:
                    param_state['cum_grad'].add_(d_p, alpha = local_lr)
                
                p.data.add_(d_p, alpha = -local_lr)
        
        self.local_steps += 1

        return loss

    def average(self, global_optim, E, global_steps, val_loss, train_loss, weight=0, tau_eff=0):
        #if weight == 0:
        #    weight = self.ratio

        w = self._get_weight(global_optim, E, global_steps, val_loss, train_loss)
        sum_w = torch.tensor(w, dtype=float)
        dist.all_reduce(sum_w, op=dist.ReduceOp.SUM)
        if sum_w.item()==0:
            weight = self.ratio
        else:
            weight = w/sum_w.item()        

        param_list = []
        for group in self.param_groups:
            for p in group['params']:
                p.data.mul_(weight)
                param_list.append(p.data)
                dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
                
        #communicate(param_list, dist.all_reduce)

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['old_init'] = torch.clone(p.data).detach()
                # Reinitialize momentum buffer
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].zero_()
                param_state['cum_grad'].zero_()        
        self.rounds += 1


    @torch.no_grad()
    def _get_weight(self, global_optim, E, global_steps, val_loss, train_loss):
        generalize = 0.0
        local_steps   = self.local_steps
        for local_group, global_group in zip(self.param_groups, global_optim.param_groups):
            for local_p, global_p in zip(local_group['params'], global_group['params']):
                if local_p.grad is None:
                    continue
                local_g  = self.state[local_p]['cum_grad']
                global_g = torch.clone(global_p.grad.data).detach()
                generalize += torch.mul(local_g, global_g).sum().item()
    
        overfitting = 0.0
        for local_group, global_group in zip(self.param_groups, global_optim.param_groups):
            for local_p, global_p in zip(local_group['params'], global_group['params']):
                if local_p.grad is None:
                    continue
                local_g  = self.state[local_p]['cum_grad']/local_steps
                global_g = torch.clone(global_p.grad.data).detach()/global_steps
                diff     = local_g - global_g
                overfitting += torch.mul(local_g, diff).sum().item()
        self.overfit += overfitting**2
    
        if generalize < 0:
            generalize = 0.0
        self.weight += generalize

        return self.weight/self.overfit