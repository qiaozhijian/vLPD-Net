# Author: Zhijian Qiao
# Shanghai Jiao Tong University
# Code adapted from PointNetVlad code: https://github.com/jac99/MinkLoc3D.git


import configparser
import os
import time

import numpy as np
from easydict import EasyDict as edict

from misc.log import log_string


class ModelParams:
    def __init__(self, model_params_path):
        config = configparser.ConfigParser()
        config.read(model_params_path)
        params = config['MODEL']

        self.model_params_path = model_params_path
        self.model = params.get('model')
        self.mink_model = params.get('mink_model', self.model)
        self.output_dim = params.getint('output_dim', 256)  # Size of the final descriptor
        self.separa_mode = params.get('separa_mode', 'KL')

        # Add gating as the last step
        if 'vlad' in self.model.lower():
            self.cluster_size = params.getint('cluster_size', 64)  # Size of NetVLAD cluster
            self.gating = params.getboolean('gating', True)  # Use gating after the NetVlad

        if 'lpd' in self.model.lower():
            self.xyz_stn = params.getboolean('xyz_stn', True)
            self.skip = params.getboolean('skip', False)

        if 'fcgf' in self.model.lower():
            self.bn_momentum = params.getfloat('bn_momentum', 0.05)
            self.normalize_feature = params.getboolean('normalize_feature', True)
        self.vcr_model = params.get('vcr_model', None)
        #######################################################################
        # Model dependent
        #######################################################################
        # Models using MinkowskiEngine
        self.mink_quantization_size = params.getfloat('mink_quantization_size')
        # Size of the local features from backbone network (only for MinkNet based models)
        # For PointNet-based models we always use 1024 intermediary features
        self.feature_size = params.getint('feature_size', 256)
        if 'planes' in params:
            self.planes = [int(e) for e in params['planes'].split(',')]
        else:
            self.planes = [32, 64, 64]

        if 'layers' in params:
            self.layers = [int(e) for e in params['layers'].split(',')]
        else:
            self.layers = [1, 1, 1]

        self.num_top_down = params.getint('num_top_down', 1)
        self.conv0_kernel_size = params.getint('conv0_kernel_size', 5)

        self.backbone = params.get('backbone', 'ResUNetBN2C')

        self.block = params.get('block', 'BasicBlock')

        if config.has_section('LPD'):
            params = config['LPD']
            self.emb_dims = params.getint('emb_dims', 512)
            self.featnet = params.get("featnet", 'MinkReg')
            if 'lpd_channels' in params:
                self.lpd_channels = [int(e) for e in params['lpd_channels'].split(',')]
            else:
                self.lpd_channels = [64, 64, 128]

        if config.has_section('REGRESS'):
            params = config['REGRESS']
            self.reg_model = params.get('model', 'MinkReg')
            self.reg_loss = params.get('loss', 'EMD')
            self.vote = params.get('vote', 'all')
            self.max_iter = params.getint('max_iter', 1)
            self.ff_dims = params.getint('ff_dims', 512)
            self.reg_model_path = params.get('model_path', "")

        if config.has_section('LPD'):
            self.lpd_pre = True
            params = config['LPD']
            self.lpd_feature_gen = [int(e) for e in params.get('feature_gen', '64,64').split(',')]
            self.lpd_knn_points = params.getint('knn_points', 20)
            self.lpd_feature_gen_second = [int(e) for e in params.get('feature_gen_second', '128,128').split(',')]
            self.lpd_feature_gen_second_num = params.getint('feature_gen_second_num', 2)
            self.lpd_feature_cat_no_extend = params.getboolean('feature_cat_no_extend', False)
            self.lpd_pooling = params.get('pooling', 'max')
            self.lpd_mid_features = [int(e) for e in params.get('mid_features', '0').split(',')]
            self.lpd_mid_features_pooling = params.get('mid_features_pooling', 'max')
            self.lpd_pooling_gem_p = params.getfloat('pooling_gem_p', 3)
            self.lpd_pooling_gem_eps = params.getfloat('pooling_gem_eps', 1e-6)
            self.lpd_se_block = params.getboolean('se_block', False)
            self.lpd_se_block_r = params.getint('se_block_r', 4)
            self.lpd_mink_backbone = params.get('mink_backbone', 'MinkLoc3d')
        else:
            self.lpd_pre = False

    def print(self):
        log_string('Model parameters:')
        param_dict = vars(self)
        for e in param_dict:
            log_string('{}: {}'.format(e, param_dict[e]))


def get_datetime():
    return time.strftime("%Y%m%d_%H%M")


def xyz_from_depth(depth_image, depth_intrinsic, depth_scale=1000.):
    # Return X, Y, Z coordinates from a depth map.
    # This mimics OpenCV cv2.rgbd.depthTo3d() function
    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]
    # Construct (y, x) array with pixel coordinates
    y, x = np.meshgrid(range(depth_image.shape[0]), range(depth_image.shape[1]), sparse=False, indexing='ij')

    X = (x - cx) * depth_image / (fx * depth_scale)
    Y = (y - cy) * depth_image / (fy * depth_scale)
    xyz = np.stack([X, Y, depth_image / depth_scale], axis=2)
    xyz[depth_image == 0] = np.nan
    return xyz


class MinkLocParams:
    """
    Params for training MinkPool models on Oxford dataset
    """

    def __init__(self, params_path, model_params_path):
        """
        Configuration files
        :param path: General configuration file
        :param model_params: Model-specific configuration
        """

        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        assert os.path.exists(model_params_path), 'Cannot find model-specific configuration file: {}'.format(
            model_params_path)
        self.params_path = params_path
        self.model_params_path = model_params_path

        config = configparser.ConfigParser()

        config.read(self.params_path)
        params = config['DEFAULT']
        self.num_points = params.getint('num_points', 4096)
        self.dataset_folder = params.get('dataset_folder')
        self.queries_folder = params.get('queries_folder')
        self.model = params.get('model')

        params = config['TRAIN']
        self.num_workers = params.getint('num_workers', 0)
        self.batch_size = params.getint('batch_size', 128)

        # Set batch_expansion_th to turn on dynamic batch sizing
        # When number of non-zero triplets falls below batch_expansion_th, expand batch size
        self.batch_expansion_th = params.getfloat('batch_expansion_th', None)
        if self.batch_expansion_th is not None:
            assert 0. < self.batch_expansion_th < 1., 'batch_expansion_th must be between 0 and 1'
            self.batch_size_limit = params.getint('batch_size_limit', 256)
            # Batch size expansion rate
            self.batch_expansion_rate = params.getfloat('batch_expansion_rate', 1.5)
            assert self.batch_expansion_rate > 1., 'batch_expansion_rate must be greater than 1'
        else:
            self.batch_size_limit = self.batch_size
            self.batch_expansion_rate = None

        self.lr = params.getfloat('lr', 1e-3)

        self.scheduler = params.get('scheduler', 'MultiStepLR')
        if self.scheduler is not None:
            if self.scheduler == 'CosineAnnealingLR':
                self.min_lr = params.getfloat('min_lr')
            elif self.scheduler == 'MultiStepLR':
                scheduler_milestones = params.get('scheduler_milestones')
                self.scheduler_milestones = [int(e) for e in scheduler_milestones.split(',')]
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(self.scheduler))

        self.epochs = params.getint('epochs', 20)

        self.weight_decay = params.getfloat('weight_decay', None)
        self.normalize_embeddings = params.getboolean('normalize_embeddings',
                                                      True)  # Normalize embeddings during training and evaluation
        self.loss = params.get('loss')

        self.lamda = params.getfloat('lamda', 1)
        self.lamda_reg = params.getfloat('lamda_reg', 0)
        self.is_register = False if self.lamda_reg <= 0.0 else True
        self.lpd_fixed = params.getboolean('lpd_fixed', False)

        if 'Contrastive' in self.loss:
            self.pos_margin = params.getfloat('pos_margin', 0.2)
            self.neg_margin = params.getfloat('neg_margin', 0.65)
        elif 'Triplet' in self.loss:
            self.margin = params.getfloat('margin', 0.4)  # Margin used in loss function
            self.swap = params.getboolean('swap', False)
        elif 'LocQuat' in self.loss:
            pass
        else:
            raise 'Unsupported loss function: {}'.format(self.loss)

        self.aug_mode = params.getint('aug_mode', 1)  # Augmentation mode (1 is default)

        self.train_file = params.get('train_file')
        self.val_file = params.get('val_file', None)

        self.eval_simple = params.getboolean('eval_simple', True)
        if not self.eval_simple:
            self.eval_database_files = ['oxford_evaluation_database.pickle', 'business_evaluation_database.pickle',
                                        'residential_evaluation_database.pickle',
                                        'university_evaluation_database.pickle', 'lgsvl_evaluation_database.pickle']

            self.eval_query_files = ['oxford_evaluation_query.pickle', 'business_evaluation_query.pickle',
                                     'residential_evaluation_query.pickle', 'university_evaluation_query.pickle',
                                     'lgsvl_evaluation_query.pickle']
        else:
            # self.eval_database_files = ['oxford_evaluation_database.pickle', 'lgsvl_evaluation_database.pickle']
            # self.eval_query_files = ['oxford_evaluation_query.pickle','lgsvl_evaluation_query.pickle']
            self.eval_database_files = ['oxford_evaluation_database.pickle']
            self.eval_query_files = ['oxford_evaluation_query.pickle']

        assert len(self.eval_database_files) == len(self.eval_query_files)

        if config.has_section('DOMAIN_ADAPT'):
            self.domain_adapt = True
            params = config['DOMAIN_ADAPT']
            self.lamda_gd = params.getfloat('lamda_gd', 0)
            self.lamda_d = params.getfloat('lamda_d', 0)
            self.repeat_g = params.getint('repeat_g', 0)
            self.d_lr = params.getfloat('lr', 1e-3)
            self.d_model = params.get('model', 'FCDor')
            self.scheduler = params.get('scheduler', 'MultiStepLR')
            if self.scheduler is not None:
                if self.scheduler == 'CosineAnnealingLR':
                    self.min_lr = params.getfloat('min_lr')
                elif self.scheduler == 'MultiStepLR':
                    scheduler_milestones = params.get('scheduler_milestones')
                    self.scheduler_milestones = [int(e) for e in scheduler_milestones.split(',')]
                else:
                    raise NotImplementedError('Unsupported LR scheduler: {}'.format(self.scheduler))

            self.d_weight_decay = params.getfloat('weight_decay', None)
            self.d_loss = params.get('loss')

            self.da_train_file = params.get('train_file', None)
        else:
            self.domain_adapt = False

        if config.has_section('REGISTARTION'):
            params = config['REGISTARTION']
            self.reg = edict()
            self.reg.batch_size = params.getint('batch_size', 32)
            self.reg.epochs = params.getint('epochs', 150)
            self.reg.num_points = params.getint('num_points', 1024)
            self.reg.iter = params.getint('iter', 1)
            self.reg.overlap = params.getfloat('overlap', 1.0)

        # Read model parameters
        self.model_params = ModelParams(self.model_params_path)

        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)

    def print(self):
        log_string('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e != 'model_params':
                log_string('{}: {}'.format(e, param_dict[e]))

        self.model_params.print()
        log_string('')
