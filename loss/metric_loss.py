# Author: Zhijian Qiao
# Shanghai Jiao Tong University
# Code adapted from PointNetVlad code: https://github.com/jac99/MinkLoc3D.git


import numpy as np
import torch
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import LpDistance
from scipy.spatial.transform import Rotation

from loss.mmd_loss import MMD_loss
from misc.log import log_string

D2G = np.pi / 180.2


def make_loss(params):
    if params.loss == 'BatchHardTripletMarginLoss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        loss_fn = BatchHardTripletLossWithMasks(params.margin, params.normalize_embeddings, params.swap)
    elif params.loss == 'BatchHardContrastiveLoss':
        loss_fn = BatchHardContrastiveLossWithMasks(params.pos_margin, params.neg_margin, params.normalize_embeddings)
    elif params.loss == 'LocQuatMSELoss':
        loss_fn = BatchLocQuatLoss(torch.nn.MSELoss())
    elif params.loss == 'LocQuatL1Loss':
        loss_fn = BatchLocQuatLoss(torch.nn.L1Loss())
    elif params.loss == 'LocQuatCloudLoss':
        loss_fn = BatchLocQuatCloudLoss()
    else:
        log_string('Unknown loss: {}'.format(params.ldss))
        raise NotImplementedError
    return loss_fn


class MMDLoss:
    def __init__(self, margin, normalize_embeddings, swap):
        self.margin = margin
        self.normalize_embeddings = normalize_embeddings
        self.loss_fn = MMD_loss()
        self.distance = LpDistance(normalize_embeddings=normalize_embeddings)
        # We use triplet loss with Euclidean distance
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)
        stats = {'metric_loss': loss.detach().cpu().item(),
                 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_non_zero_triplets': self.loss_fn.reducer.triplets_past_filter,
                 'num_triplets': len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }


def quat_to_mat_torch(q, req_grad=False):
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    m11 = 1 - 2 * y * y - 2 * z * z
    m21 = 2 * x * y + 2 * w * z
    m31 = 2 * x * z - 2 * w * y
    m12 = 2 * x * y - 2 * w * z
    m22 = 1 - 2 * x * x - 1 * 2 * z * z
    m32 = 2 * y * z + 2 * w * x
    m13 = 2 * x * z + 2 * w * y
    m23 = 2 * y * z - 2 * w * x
    m33 = 1 - 2 * x * x - 2 * y * y

    return torch.tensor([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]], requires_grad=req_grad)


def to_clouds(batch, device=torch.device('cpu')):
    clouds = [[] for i in range(batch[-1][0] + 1)]
    for p in batch:
        clouds[p[0]].append(p[1:])
    clouds = [torch.Tensor(c).to(device) for c in clouds]
    return clouds


class BatchLocQuatCloudLoss:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = torch.nn.L1Loss()
        pass

    def __call__(self, batch, quat, truth_quat):
        batch = batch['coords'].detach().numpy()
        rots = []

        loss = 0.0
        raw_clouds = to_clouds(batch, self.device)
        for q, qt, cloud in zip(quat, truth_quat, raw_clouds):
            m1 = quat_to_mat_torch(q, True).to(self.device)
            m2 = quat_to_mat_torch(qt).to(self.device)

            c1 = torch.matmul(cloud, m1)
            c2 = torch.matmul(cloud, m2)

            loss += self.loss_fn(c1, c2)

        loss = loss / len(raw_clouds)
        return loss, {'loss': loss}


class BatchLocQuatLoss:
    def __init__(self, tloss):
        self.loss_fn = tloss

    def __call__(self, quats, quats_truth):
        loss = self.loss_fn(quats, quats_truth)
        rloss = np.array([0., 0., 0.])
        maxrloss = 0
        max_loss_r1 = []
        max_loss_r2 = []
        for q, t in zip(quats.detach().cpu().numpy(), quats_truth.detach().cpu().numpy()):
            r1 = Rotation.from_quat(q).as_rotvec() / D2G
            r2 = Rotation.from_quat(t).as_rotvec() / D2G
            r = abs(r1 - r2)
            rloss += r
            if maxrloss < np.mean(r):
                max_loss_r1 = r1
                max_loss_r2 = r2
        rloss = np.mean(rloss) / len(quats)
        rloss2 = np.linalg.norm(rloss) / len(quats)
        return loss, {'loss': loss.detach().cpu(),
                      'avg.rloss': rloss,
                      'avg.rloss2': rloss2,
                      'max_loss_r1_x': max_loss_r1[0],
                      'max_loss_r1_y': max_loss_r1[1],
                      'max_loss_r1_z': max_loss_r1[2],

                      'max_loss_r2_x': max_loss_r2[0],
                      'max_loss_r2_y': max_loss_r2[1],
                      'max_loss_r2_z': max_loss_r2[2], }


class HardTripletMinerWithMasks:
    # Hard triplet miner
    def __init__(self, distance):
        self.distance = distance
        # Stats
        self.max_pos_pair_dist = None
        self.max_neg_pair_dist = None
        self.mean_pos_pair_dist = None
        self.mean_neg_pair_dist = None
        self.min_pos_pair_dist = None
        self.min_neg_pair_dist = None

    def __call__(self, embeddings, positives_mask, negatives_mask):
        assert embeddings.dim() == 2
        d_embeddings = embeddings.detach()
        with torch.no_grad():
            hard_triplets = self.mine(d_embeddings, positives_mask, negatives_mask)
        return hard_triplets

    def mine(self, embeddings, positives_mask, negatives_mask):
        # Based on pytorch-metric-learning implementation n x n
        dist_mat = self.distance(embeddings)
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = get_max_per_row(dist_mat, positives_mask)
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = get_min_per_row(dist_mat, negatives_mask)
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        a = torch.arange(dist_mat.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]
        self.max_pos_pair_dist = torch.max(hardest_positive_dist).item()
        self.max_neg_pair_dist = torch.max(hardest_negative_dist).item()
        self.mean_pos_pair_dist = torch.mean(hardest_positive_dist).item()
        self.mean_neg_pair_dist = torch.mean(hardest_negative_dist).item()
        self.min_pos_pair_dist = torch.min(hardest_positive_dist).item()
        self.min_neg_pair_dist = torch.min(hardest_negative_dist).item()
        return a, p, n


def get_max_per_row(mat, mask):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone().detach()
    mat_masked[~mask] = 0
    # mat_masked2=torch.where(mask,mat,torch.zeros(mat.shape))
    # pdb.set_trace()
    return torch.max(mat_masked, dim=1), non_zero_rows


def get_min_per_row(mat, mask):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone().detach()
    mat_masked[~mask] = float('inf')
    return torch.min(mat_masked, dim=1), non_inf_rows


class BatchHardTripletLossWithMasks:
    def __init__(self, margin, normalize_embeddings, swap):
        self.margin = margin
        self.normalize_embeddings = normalize_embeddings
        self.distance = LpDistance(normalize_embeddings=normalize_embeddings)
        # We use triplet loss with Euclidean distance
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        self.loss_fn = losses.TripletMarginLoss(margin=self.margin, swap=swap, distance=self.distance)

    # 3510
    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)
        stats = {'metric_loss': loss.detach().cpu().item(),
                 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'num_non_zero_triplets': self.loss_fn.reducer.triplets_past_filter,
                 'num_triplets': len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }

        return loss, stats, hard_triplets


class BatchHardContrastiveLossWithMasks:
    def __init__(self, pos_margin, neg_margin, normalize_embeddings):
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.distance = LpDistance(normalize_embeddings=normalize_embeddings)
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        # We use contrastive loss with squared Euclidean distance
        self.loss_fn = losses.ContrastiveLoss(pos_margin=self.pos_margin, neg_margin=self.neg_margin,
                                              distance=self.distance)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)
        stats = {'metric_loss': loss.detach().cpu().item(),
                 'avg_embedding_norm': self.loss_fn.distance.final_avg_query_norm,
                 'pos_pairs_above_low': self.loss_fn.reducer.reducers['pos_loss'].pos_pairs_above_low,
                 'neg_pairs_above_low': self.loss_fn.reducer.reducers['neg_loss'].neg_pairs_above_low,
                 'pos_loss': self.loss_fn.reducer.reducers['pos_loss'].pos_loss,
                 'neg_loss': self.loss_fn.reducer.reducers['neg_loss'].neg_loss,
                 'num_pairs': 2 * len(hard_triplets[0]),
                 'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
                 'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
                 'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
                 'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
                 'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
                 'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
                 }

        return loss, stats, hard_triplets
