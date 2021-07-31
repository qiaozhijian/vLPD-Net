# Author: Zhijian Qiao
# Shanghai Jiao Tong University
# Code adapted from PointNetVlad code: https://github.com/jac99/MinkLoc3D.git

import os

import numpy as np
import torch

from misc.log import log_string
from models.discriminator.resnetD import FCDor
from models.vcrnet.vcrnet import VCRNet
from models.vlpdnet.vlpdnet import vLPDNet


def model_factory(params):
    if 'vLPDNet' in params.model_params.model:
        model = vLPDNet(params)
    else:
        raise NotImplementedError('Model not implemented: {}'.format(params.model_params.model))

    if hasattr(model, 'print_info'):
        model.print_info()

    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    log_string('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))

    if params.domain_adapt:
        if params.d_model == 'FCDor':
            d_model = FCDor(feature_dim=params.model_params.feature_size, n_layers=3, D=3)
        else:
            raise NotImplementedError('d_model not implemented: {}'.format(params.d_model))

        if hasattr(d_model, 'print_info'):
            d_model.print_info()
    else:
        d_model = None

    vcr_model = None
    if params.is_register:
        vcr_model = VCRNet(params.model_params)
        if params.model_params.reg_model_path != "":
            checkpoint_dict = torch.load(params.model_params.reg_model_path, map_location=torch.device('cpu'))
            vcr_model.load_state_dict(checkpoint_dict, strict=True)
            log_string('load vcr_model with {}'.format(params.model_params.reg_model_path))

    # Move the model to the proper device before configuring the optimizer
    if torch.cuda.is_available():
        device = "cuda"
        model.to(device)
        d_model = d_model.to(device) if params.domain_adapt else None
        vcr_model = vcr_model.to(device) if params.is_register else None
        if torch.cuda.device_count() > 1:
            vcr_model = torch.nn.DataParallel(vcr_model)
    else:
        device = "cpu"

    log_string('Model device: {}'.format(device))

    return model, device, d_model, vcr_model


def load_weights(weights, model, optimizer=None, scheduler=None):
    starting_epoch = 0

    if weights is None or weights == "":
        return starting_epoch

    if not os.path.exists(weights):
        log_string("error: checkpoint not exists!")

    checkpoint_dict = torch.load(weights, map_location=torch.device('cpu'))
    if 'state_dict' in checkpoint_dict.keys():
        saved_state_dict = checkpoint_dict['state_dict']
    else:
        saved_state_dict = checkpoint_dict

    model_state_dict = model.state_dict()  # 获取已创建net的state_dict
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_state_dict}
    model_state_dict.update(saved_state_dict)
    model.load_state_dict(model_state_dict, strict=True)

    if 'epoch' in checkpoint_dict.keys():
        starting_epoch = checkpoint_dict['epoch'] + 1

    if 'optimizer' in checkpoint_dict.keys() and optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])

    if 'scheduler' in checkpoint_dict.keys() and scheduler is not None:
        scheduler.load_state_dict(checkpoint_dict['scheduler'])

    log_string("load checkpoint " + weights + " starting_epoch: " + str(starting_epoch))
    torch.cuda.empty_cache()

    return starting_epoch
