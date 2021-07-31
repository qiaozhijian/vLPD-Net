# Author: Zhijian Qiao
# Shanghai Jiao Tong University
# Code adapted from PointNetVlad code: https://github.com/jac99/MinkLoc3D.git


import argparse
import os
import sys

import torch

sys.path.append(os.path.join(os.path.abspath("./"), "../"))
from training.trainer import Trainer
from misc.utils import MinkLocParams
from misc.log import log_string
from dataloader.dataset_utils import make_dataloaders
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Minkowski Net embeddings using BatchHard negative mining')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--qzj_debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--checkpoint', type=str, required=False, help='Trained model weights', default="")
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)

    args = parser.parse_args()
    log_string('Training config path: {}'.format(args.config))
    log_string('Model config path: {}'.format(args.model_config))
    log_string('Debug mode: {}'.format(args.debug))
    log_string('Visualize: {}'.format(args.visualize))

    params = MinkLocParams(args.config, args.model_config)
    if args.qzj_debug:
        params.batch_size = 4
        if params.is_register:
            params.reg.batch_size = 4
    params.print()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    dataloaders = make_dataloaders(params, debug=args.debug)
    trainer = Trainer(dataloaders, params, debug=args.debug, visualize=args.visualize, checkpoint=args.checkpoint)
    if args.debug:
        pdb.set_trace()

    trainer.do_train()
