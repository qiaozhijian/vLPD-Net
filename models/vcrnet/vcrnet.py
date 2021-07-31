import torch
import torch.nn as nn
import torch.nn.functional as F

from misc.utils import ModelParams
from models.vcrnet.transformer import Transformer
from models.vlpdnet.lpdnet_model import LPDNet, LPDNetOrign


class PoseSolver(nn.Module):
    def __init__(self, model_params, origin=False):
        super(PoseSolver, self).__init__()

        self.pointer = Transformer(model_params=model_params)

        self.head = EPCOR(model_params=model_params)

        self.svd = SVD_Weighted()

        self.loss = torch.nn.functional.mse_loss

        self.origin = origin

    def forward(self, src, tgt, src_embedding, tgt_embedding, positive_T, svd=True):
        # input: #[B,1,num,3] [B,1,num,3] [B,1,C,num,1] [B,posi_num,C,num,1]
        # expected: [B,C,num]
        batch, posi_num, num_points, C = tgt.size()
        # src = src.repeat(1,posi_num,1,1).transpose(-2,-1).contiguous().view(-1,C,num_points)
        src = src.transpose(-2, -1).contiguous().view(-1, C, num_points)
        tgt = tgt.transpose(-2, -1).contiguous().view(-1, C, num_points)
        C = tgt_embedding.size(2)
        src_embedding = src_embedding.squeeze(-1).repeat(1, posi_num, 1, 1).contiguous().view(-1, C, num_points)
        tgt_embedding = tgt_embedding.squeeze(-1).contiguous().view(-1, C, num_points)

        src_embedding_p, tgt_embedding_p = self.pointer(src_embedding, tgt_embedding)
        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        src_corr, src_weight, outlier_src_mask, mask_tgt = self.head(src_embedding, tgt_embedding, src, tgt)

        if svd:
            with torch.no_grad():
                rotation_ab_pred, translation_ab_pred = self.svd(src, src_corr, src_weight)

        if self.origin:
            b, _, num = tgt.size()
            mask = outlier_src_mask.contiguous().view(b, 1, num).repeat(1, 3, 1)
            srcK = torch.masked_fill(src, mask, 0)
            src_corrK = torch.masked_fill(src_corr, mask, 0)
            return {'rotation_ab_pred': rotation_ab_pred, 'translation_ab_pred': translation_ab_pred,
                    'srcK': srcK, 'src_corrK': src_corrK}

        if positive_T is None:
            return rotation_ab_pred, translation_ab_pred

        if svd:
            loss = self.cor_loss(src, src_corr, outlier_src_mask, positive_T)

            rotation_ab, translation_ab, loss_pose = self.pose_loss(rotation_ab_pred, translation_ab_pred, positive_T)

            return {'rotation_ab_pred': rotation_ab_pred, 'translation_ab_pred': translation_ab_pred,
                    'rotation_ab': rotation_ab, 'translation_ab': translation_ab,
                    'loss_point': loss, 'loss_pose': loss_pose, 'mask_tgt': mask_tgt}
        else:
            return {'rotation_ab_pred': None, 'translation_ab_pred': None,
                    'rotation_ab': None, 'translation_ab': None,
                    'loss_point': None, 'loss_pose': None, 'mask_tgt': mask_tgt}

    def pose_loss(self, rotation_ab_pred, translation_ab_pred, gt_T):

        gt_T = gt_T.view(-1, 4, 4)
        rotation_ab = gt_T[:, :3, :3]
        translation_ab = gt_T[:, :3, 3]
        batch_size = translation_ab.size(0)
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        loss_pose = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                    + F.mse_loss(translation_ab_pred, translation_ab)
        return rotation_ab, translation_ab, loss_pose

    def cor_loss(self, src, src_corr, outlier_src_mask, gt_T):

        gt_T = gt_T.view(-1, 4, 4)
        rotation_ab = gt_T[:, :3, :3]
        translation_ab = gt_T[:, :3, 3]

        transformed_srcK = torch.matmul(rotation_ab, src) + translation_ab.unsqueeze(2)

        b, _, num = transformed_srcK.size()
        mask = outlier_src_mask.contiguous().view(b, 1, num).repeat(1, 3, 1)

        transformed_srcK = torch.masked_fill(transformed_srcK, mask, 0)
        src_corrK = torch.masked_fill(src_corr, mask, 0)

        loss = self.loss(transformed_srcK, src_corrK)

        return loss


class VCRNet(nn.Module):
    def __init__(self, model_params: ModelParams):
        super(VCRNet, self).__init__()
        if model_params.featnet == "lpdnet":
            self.emb_nn = LPDNet(emb_dims=model_params.emb_dims, channels=model_params.lpd_channels)
        elif model_params.featnet.lower() == "lpdnetorigin":
            self.emb_nn = LPDNetOrign(emb_dims=model_params.emb_dims, channels=model_params.lpd_channels)
        else:
            print("featnet error")
        self.solver = PoseSolver(model_params, origin=True)

    def forward(self, *input):
        src = input[0]
        tgt = input[1]

        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        src = src.unsqueeze(1)
        tgt = tgt.unsqueeze(1)
        src_embedding = src_embedding.unsqueeze(1)
        tgt_embedding = tgt_embedding.unsqueeze(1)

        # input: #[B,1,num,3] [B,posi_num,num,3] [B,1,C,num,1] [B,posi_num,C,num,1]
        reg_dict = self.solver(src, tgt, src_embedding, tgt_embedding, None)
        rotation_ab = reg_dict['rotation_ab_pred']
        translation_ab = reg_dict['translation_ab_pred']
        rotation_ba = rotation_ab.transpose(2, 1).contiguous()
        translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)
        srcK = reg_dict['srcK']
        src_corrK = reg_dict['src_corrK']

        return srcK, src_corrK, rotation_ab, translation_ab, rotation_ba, translation_ba


class SVD_Weighted(nn.Module):
    def __init__(self):
        super(SVD_Weighted, self).__init__()

    def forward(self, src, src_corr, weights: torch.Tensor = None):
        _EPS = 1e-5  # To prevent division by zero
        a = src.transpose(-1, -2)
        b = src_corr.transpose(-1, -2)
        batch_size, num_src, dims = a.size()

        if weights is None:
            weights = torch.ones(size=(batch_size, num_src), device=torch.device('cuda')) / num_src

        """Compute rigid transforms between two point sets

        Args:
            a (torch.Tensor): (B, M, 3) points
            b (torch.Tensor): (B, N, 3) points
            weights (torch.Tensor): (B, M)

        Returns:
            Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
        """

        weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
        centroid_a = torch.sum(a * weights_normalized, dim=1)
        centroid_b = torch.sum(b * weights_normalized, dim=1)
        a_centered = a - centroid_a[:, None, :]
        b_centered = b - centroid_b[:, None, :]
        cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

        # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
        # and choose based on determinant to avoid flips
        u, s, v = torch.svd(cov, some=False, compute_uv=True)
        rot_mat_pos = v @ u.transpose(-1, -2)
        v_neg = v.clone()
        v_neg[:, :, 2] *= -1
        rot_mat_neg = v_neg @ u.transpose(-1, -2)
        rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
        assert torch.all(torch.det(rot_mat) > 0)

        # Compute translation (uncenter centroid)
        translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

        return rot_mat, translation.view(batch_size, 3)


class EPCOR(nn.Module):
    """Generate the VCP points based on K most similar points

    """

    def __init__(self, model_params):
        super(EPCOR, self).__init__()
        self.model_params = model_params

    def forward(self, *input):

        src_emb = input[0]
        tgt_emb = input[1]
        src = input[2]
        tgt = input[3]

        if self.training:
            src_corr, src_weight, outlier_src_mask, mask_tgt = self.getCopairALL(src, src_emb, tgt, tgt_emb)
        else:
            src_corr, src_weight, outlier_src_mask, mask_tgt = self.selectCom_adap(src, src_emb, tgt, tgt_emb)

        return src_corr, src_weight, outlier_src_mask, mask_tgt

    def get_sparse_w(self, pairwise_distance, tau, K):
        batch_size, num_points, num_points_t = pairwise_distance.size()
        tgt_K = int(num_points_t * tau)
        src_K = int(num_points * tau)

        scoresSoftCol = torch.softmax(pairwise_distance, dim=2)  # [b,num,num]
        scoresColSum = torch.sum(scoresSoftCol, dim=1, keepdim=True)
        topmin = scoresColSum.topk(k=tgt_K, dim=-1, largest=False)[0][..., -1].unsqueeze(-1)
        mask_tgt = scoresColSum < topmin
        # a = torch.sum(mask_tgt, dim=-1)

        scoresSoftRow = torch.softmax(pairwise_distance, dim=1)  # [b,num,num]
        scoresRowSum = torch.sum(scoresSoftRow, dim=2, keepdim=True)
        topmin = scoresRowSum.topk(k=src_K, dim=1, largest=False)[0][:, -1, :].unsqueeze(-1)
        mask_src = scoresRowSum < topmin
        # mask_src = scoresRowSum < tau

        mask = mask_src + mask_tgt

        s = (torch.sum(mask_src, dim=-2) + torch.sum(mask_tgt, dim=-1)) / torch.full(size=(batch_size, 1),
                                                                                     fill_value=num_points + num_points_t).to(
            mask_tgt.device).type_as(pairwise_distance)

        topk = scoresSoftCol.topk(k=K, dim=2)[0][..., -1].unsqueeze(-1)
        mask = mask + scoresSoftCol < topk

        src_tgt_weight_sparse = torch.masked_fill(scoresSoftCol, mask, 0)
        scoresColSum_sparse = torch.sum(src_tgt_weight_sparse, dim=-1, keepdim=True)  # [B, num, 1]
        scoresColSum_sparse = torch.masked_fill(scoresColSum_sparse, scoresColSum_sparse < 1e-5, 1e-5)  # 防止除以0
        src_tgt_weight = torch.div(src_tgt_weight_sparse, scoresColSum_sparse)

        # src_tgt_weight = torch.softmax(src_tgt_weight, dim=2)

        # 计算
        val_sum = torch.sum(src_tgt_weight, dim=-1, keepdim=True)  # [B, num, 1]
        val_sum_inlier = torch.masked_fill(val_sum, mask_src, 0.0).squeeze(-1)
        sum_ = torch.sum(val_sum_inlier, dim=[-1], keepdim=True)
        src_weight = torch.div(val_sum_inlier, sum_)

        return s, src_weight, mask_src, mask_tgt, src_tgt_weight

    def selectCom_adap(self, src, src_emb, tgt, tgt_emb, tau=0.3):

        batch_size, _, num_points = src.size()
        batch_size, _, num_points_t = tgt.size()

        inner = -2 * torch.matmul(src_emb.transpose(2, 1).contiguous(), tgt_emb)
        xx = torch.sum(src_emb ** 2, dim=1, keepdim=True).transpose(2, 1).contiguous()
        yy = torch.sum(tgt_emb ** 2, dim=1, keepdim=True)

        pairwise_distance = -xx - inner
        src_tgt_weight = pairwise_distance - yy

        s, src_weight, mask_src, mask_tgt, src_tgt_weight = self.get_sparse_w(src_tgt_weight, tau, K=1)

        src_corr = torch.matmul(tgt, src_tgt_weight.transpose(2, 1).contiguous())

        return src_corr, src_weight, mask_src, mask_tgt

    def getCopairALL(self, src, src_emb, tgt, tgt_emb):

        batch_size, n_dims, num_points = src.size()
        # Calculate the distance matrix
        inner = -2 * torch.matmul(src_emb.transpose(2, 1).contiguous(), tgt_emb)
        xx = torch.sum(src_emb ** 2, dim=1, keepdim=True).transpose(2, 1).contiguous()
        yy = torch.sum(tgt_emb ** 2, dim=1, keepdim=True)

        pairwise_distance = -xx - inner
        pairwise_distance = pairwise_distance - yy

        scores = torch.softmax(pairwise_distance, dim=2)  # [b,num,num]
        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

        src_weight = torch.ones(size=(batch_size, num_points), device=src_corr.device) / num_points

        outlier_src_mask = torch.full((batch_size, num_points, 1), False, device=src_corr.device, dtype=torch.bool)
        mask_tgt = torch.full((batch_size, num_points, 1), False, device=src_corr.device, dtype=torch.bool)

        return src_corr, src_weight, outlier_src_mask, mask_tgt

    def return_mask(self):
        return self.mask_src, self.mask_tgt
