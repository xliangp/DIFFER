import torch


def make_optimizer(cfg, model, center_criterion,lr_new=0):
    params = []
    large_lr_layers = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if lr_new==0:
            lr = cfg.SOLVER.BASE_LR
        else:
            lr=lr_new
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            #if "classifier" in key or "arcface" in key:
            if "head_cls" in key:
                large_lr_layers.append(key)
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.LARGE_FC_FACTOR

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        
    print(f'Using {cfg.SOLVER.LARGE_FC_FACTOR} times learning rate for ',large_lr_layers)

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center
