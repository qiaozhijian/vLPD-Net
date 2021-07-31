from __future__ import print_function

import math

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from models.minkloc3d.minkpool import MinkPool
from models.vcrnet.vcrnet import PoseSolver
from models.vlpdnet.lpdnet_model import LPDNet, LPDNetOrign


class vLPDNet(nn.Module):
    def __init__(self, params):
        super(vLPDNet, self).__init__()
        self.params = params
        model_params = params.model_params
        self.featnet = model_params.featnet
        if model_params.featnet == "lpdnet":
            self.emb_nn = LPDNet(emb_dims=model_params.emb_dims, channels=model_params.lpd_channels)
        elif model_params.featnet.lower() == "lpdnetorigin":
            self.emb_nn = LPDNetOrign(emb_dims=model_params.emb_dims, channels=model_params.lpd_channels)
        else:
            print("featnet error")
        if params.lpd_fixed:
            self.emb_nn.requires_grad = False
            self.emb_nn.eval()
        # self.net_vlad = NetVLADLoupe(feature_size=args.emb_dims, cluster_size=64,
        #                              output_dim=args.output_dim, gating=True, add_batch_norm=True,
        #                              is_training=True)
        self.net_vlad = MinkPool(model_params, in_channels=model_params.emb_dims)

        self.is_register = params.is_register
        # self.is_register = True
        if params.domain_adapt:
            self.is_register = False
            self.params.lpd_fixed = False
        if params.is_register:
            self.pose_solver = PoseSolver(model_params=model_params)
        else:
            self.pose_solver = None

    # input x [B,1,N,3]
    # intermediate feat_x [B,C,N,1]
    # output g_feat [N,3]
    def forward(self, source_batch=None, target_batch=None, gt_T=None):
        if self.params.lpd_fixed:
            self.emb_nn.eval()
            with torch.no_grad():
                if self.is_register and source_batch != None:
                    source = source_batch["cloud"].unsqueeze(1)
                    feat_x_s = self.emb_nn(source)
                target = target_batch["cloud"].unsqueeze(1)
                feat_x_t = self.emb_nn(target)
        else:
            if self.is_register and source_batch != None:
                source = source_batch["cloud"].unsqueeze(1)
                feat_x_s = self.emb_nn(source)
            target = target_batch["cloud"].unsqueeze(1)
            feat_x_t = self.emb_nn(target)

        # registration
        # batch_R, batch_t, transformed_xyz = self.reg_model(feat_x_s, feat_x_t, source)
        if self.is_register and source_batch != None:
            reg_dict = self.pose_solver(source, target, feat_x_s.unsqueeze(1), feat_x_t.unsqueeze(1), gt_T, svd=False)
            fs = []
            pcls = []
            mask_tgt = reg_dict['mask_tgt'].squeeze(-1)
            for i in range(mask_tgt.shape[0]):
                mask = ~mask_tgt[i]
                # a = torch.sum(mask)
                fs.append(feat_x_t[i][:, mask, :])
                pcls.append(target[i][:, mask, :])
        else:
            reg_dict = None
            fs = []
            pcls = []
            for i in range(feat_x_t.shape[0]):
                fs.append(feat_x_t[i])
                pcls.append(target[i])

        # g_fea = self.net_vlad(feat_x_t, target)
        g_fea = self.net_vlad(fs, pcls)

        if isinstance(g_fea, dict):
            g_fea = g_fea['g_fea']

        return {
            'embeddings': g_fea,
            'reg_dict': reg_dict
        }

        return {'g_fea': g_fea, 'l_fea': feat_x_t, 'node_fea': None}

    def registration(self, src, tgt, src_embedding, tgt_embedding, gt_T):

        reg_dict = self.pose_solver(src, tgt, src_embedding, tgt_embedding, gt_T)

        return reg_dict

    def registration_only(self, src, tgt, gt_T):
        # input x [B,1,N,3]
        src_embedding = self.emb_nn(src)
        src_embedding = src_embedding.permute(0, 3, 1, 2)
        if self.is_register:
            tgt_embedding = self.emb_nn(tgt)
            tgt_embedding = tgt_embedding.permute(0, 3, 1, 2)
        reg_dict = self.pose_solver(src, tgt, src_embedding, tgt_embedding, gt_T)
        return reg_dict


class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(
            feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(
            1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(
                cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        self.weight_initialization_all()

    # [B, dims, num_points, 1]
    def forward(self, x, coord=None):
        max_samples = x.size(2)  # num of points
        x = x.transpose(1, 3).contiguous()
        x = x.view((-1, max_samples, self.feature_size))
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1,
                                         max_samples, self.cluster_size)
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)
        activation = activation.view((-1, max_samples, self.cluster_size))

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a

        vlad = F.normalize(vlad, dim=1, p=2)
        # print(vlad.shape,self.cluster_size,self.feature_size,self.cluster_size * self.feature_size)
        vlad = vlad.reshape((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights)

        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad

    def weight_initialization_all(self):
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for m in self.modules():
                    self.weight_init(m)
            else:
                self.weight_init(m)

    def weight_init(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

        self.weight_initialization_all()

    def weight_initialization_all(self):
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for m in self.modules():
                    self.weight_init(m)
            else:
                self.weight_init(m)

    def weight_init(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            if gates.size(0) == 1:
                gates = gates
            else:
                gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation


def extract_features(model,
                     xyz,
                     rgb=None,
                     normal=None,
                     voxel_size=0.05,
                     device=None,
                     skip_check=False,
                     is_eval=True):
    '''
    xyz is a N x 3 matrix
    rgb is a N x 3 matrix and all color must range from [0, 1] or None
    normal is a N x 3 matrix and all normal range from [-1, 1] or None

    if both rgb and normal are None, we use Nx1 one vector as an input

    if device is None, it tries to use gpu by default

    if skip_check is True, skip rigorous checks to speed up

    model = model.to(device)
    xyz, feats = extract_features(model, xyz)
    '''
    if is_eval:
        model.eval()

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feats = []
    if rgb is not None:
        # [0, 1]
        feats.append(rgb - 0.5)

    if normal is not None:
        # [-1, 1]
        feats.append(normal / 2)

    if rgb is None and normal is None:
        feats.append(np.ones((len(xyz), 1)))

    feats = np.hstack(feats)

    # Voxelize xyz and feats
    coords = np.floor(xyz / voxel_size)

    coords, inds = ME.utils.sparse_quantize(coords, return_index=True)

    # Convert to batched coords compatible with ME
    coords = ME.utils.batched_coordinates([coords])
    return_coords = xyz[inds]

    feats = feats[inds]

    feats = torch.tensor(feats, dtype=torch.float32)
    coords = coords.clone().detach()

    stensor = ME.SparseTensor(feats, coordinates=coords, device=device)

    return return_coords, model(stensor).F
