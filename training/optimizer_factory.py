import torch

from misc.utils import MinkLocParams


def optimizer_factory(params: MinkLocParams, model, d_model):
    # Training elements
    if params.weight_decay is None or params.weight_decay == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
        optimizer_d = torch.optim.Adam(d_model.parameters(), lr=params.d_lr) if params.domain_adapt else None
        print("optimizer use Adam with lr {}".format(params.lr))
        if optimizer_d is not None:
            print("optimizer_d use Adam with and lr {}".format(params.d_lr))
    else:
        if params.lpd_fixed:
            ignored_params = list(map(id, model.emb_nn.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': model.emb_nn.parameters(), 'lr': 0.0}], lr=params.lr, weight_decay=params.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        optimizer_d = torch.optim.Adam(d_model.parameters(), lr=params.d_lr,
                                       weight_decay=params.d_weight_decay) if params.domain_adapt else None
        print("optimizer use Adam with weight_decay {} and lr {}".format(params.weight_decay, params.lr))
        if optimizer_d is not None:
            print("optimizer_d use Adam with weight_decay {} and lr {}".format(params.d_weight_decay, params.d_lr))

    return optimizer, optimizer_d


def scheduler_factory(params: MinkLocParams, optimizer, optimizer_d=None):
    # Training elements
    optimizer_d = optimizer if optimizer_d == None else optimizer_d
    if params.scheduler is None:
        scheduler = None
        scheduler_d = None
    else:
        if params.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs + 1,
                                                                   eta_min=params.min_lr)
            scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=params.epochs + 1,
                                                                     eta_min=params.min_lr) if params.domain_adapt else None
        elif params.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.scheduler_milestones, gamma=0.1)
            scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, params.scheduler_milestones,
                                                               gamma=0.1) if params.domain_adapt else None
        else:
            scheduler = None
            scheduler_d = None
            raise NotImplementedError('Unsupported LR scheduler: {}'.format(params.scheduler))

    return scheduler, scheduler_d
