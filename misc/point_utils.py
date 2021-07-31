import numpy as np
import torch
from scipy.spatial.transform import Rotation


def group(x: torch.FloatTensor, idx: torch.LongTensor):
    """
    :param  x:      (B, N, F)
    :param  idx:    (B, M, k)
    :return (B, M, k, F)
    """
    B, N, F = tuple(x.size())
    _, M, k = tuple(idx.size())

    x = x.unsqueeze(1).expand(B, M, N, F)
    idx = idx.unsqueeze(3).expand(B, M, k, F)

    return torch.gather(x, dim=2, index=idx)


def get_knn_idx_dist(pos: torch.FloatTensor, query: torch.FloatTensor, k, offset=0):
    """
    :param  pos:     (B, N, F)
    :param  query:   (B, M, F)
    :return knn_idx: (B, M, k)
    """
    B, N, F = tuple(pos.size())
    M = query.size(1)

    pos = pos.unsqueeze(1).expand(B, M, N, F)
    query = query.unsqueeze(2).expand(B, M, N, F)  # B * M * N * F
    dist = torch.sum((pos - query) ** 2, dim=3, keepdim=False)  # B * M * N
    knn_idx = torch.argsort(dist, dim=2)[:, :, offset:k + offset]  # B * M * k
    knn_dist = torch.gather(dist, dim=2, index=knn_idx)  # B * M * k

    return knn_idx, knn_dist


def gather(x: torch.FloatTensor, idx: torch.LongTensor):
    """
    :param  x:      (B, N, F)
    :param  idx:    (B, M)
    :return (B, M, F)
    """
    # x       : B * N * F
    # idx     : B * M
    # returns : B * M * F
    B, N, F = tuple(x.size())
    _, M = tuple(idx.size())

    idx = idx.unsqueeze(2).expand(B, M, F)

    return torch.gather(x, dim=1, index=idx)


def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, C, N]/[B,C,N,1]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, C, S]
    """
    if len(points.shape) == 4:
        points = points.squeeze()
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    points = points.permute(0, 2, 1)  # (B,N,C)
    new_points = points[batch_indices, idx, :]
    if len(new_points.shape) == 3:
        new_points = new_points.permute(0, 2, 1)
    elif len(new_points.shape) == 4:
        new_points = new_points.permute(0, 3, 1, 2)
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, C, N]
        new_xyz: query points, [B, C, S]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, C, N = xyz.shape
    _, _, S = new_xyz.shape
    sqrdists = square_distance(new_xyz, xyz)
    if radius is not None:
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
    else:
        group_idx = torch.sort(sqrdists, dim=-1)[1][:, :, :nsample]
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, C, N]
        dst: target points, [B, C, M]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, _, N = src.shape
    _, _, M = dst.shape
    dist = -2 * torch.matmul(src.permute(0, 2, 1), dst)
    dist += torch.sum(src ** 2, 1).view(B, N, 1)
    dist += torch.sum(dst ** 2, 1).view(B, 1, M)
    return dist


def upsample_inter(xyz1, xyz2, points1, points2, k):
    """
    Input:
        xyz1: input points position data, [B, C, N]
        xyz2: sampled input points position data, [B, C, S]
        points1: input points data, [B, D, N]/[B,D,N,1]
        points2: input points data, [B, D, S]/[B,D,S,1]
        k:
    Return:
        new_points: upsampled points data, [B, D+D, N]
    """
    if points1 is not None:
        if len(points1.shape) == 4:
            points1 = points1.squeeze()
    if len(points2.shape) == 4:
        points2 = points2.squeeze()
    B, C, N = xyz1.size()
    _, _, S = xyz2.size()

    dists = square_distance(xyz1, xyz2)  # (B, N, S)
    dists, idx = dists.sort(dim=-1)
    dists, idx = dists[:, :, :k], idx[:, :, :k]  # [B, N, 3]
    dists[dists < 1e-10] = 1e-10
    weight = 1.0 / dists  # [B, N, 3]
    weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]; weight = [64, 1024, 3]
    interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, 1, N, k),
                                    dim=3)  # (B,D,N); idx = [64, 1024, 3]; points2 = [64, 64, 64];
    if points1 is not None:
        new_points = torch.cat([points1, interpolated_points], dim=1)  # points1 = [64, 64, 1024];
        return new_points
    else:
        return interpolated_points


def pairwise_distance(x):
    batch_size = x.size(0)
    point_cloud = torch.squeeze(x)
    if batch_size == 1:
        point_cloud = torch.unsqueeze(point_cloud, 0)
    point_cloud_transpose = torch.transpose(point_cloud, dim0=1, dim1=2)
    point_cloud_inner = torch.matmul(point_cloud_transpose, point_cloud)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = torch.sum(point_cloud ** 2, dim=1, keepdim=True)
    point_cloud_square_transpose = torch.transpose(point_cloud_square, dim0=1, dim1=2)
    return point_cloud_square + point_cloud_inner + point_cloud_square_transpose


def gather_neighbor(x, nn_idx, n_neighbor):
    x = torch.squeeze(x)
    batch_size = x.size()[0]
    num_dim = x.size()[1]
    num_point = x.size()[2]
    point_expand = x.unsqueeze(2).expand(batch_size, num_dim, num_point, num_point)
    nn_idx_expand = nn_idx.unsqueeze(1).expand(batch_size, num_dim, num_point, n_neighbor)
    pc_n = torch.gather(point_expand, -1, nn_idx_expand)
    return pc_n


def get_neighbor_feature(x, n_point, n_neighbor):
    if len(x.size()) == 3:
        x = x.unsqueeze()
    adj_matrix = pairwise_distance(x)
    _, nn_idx = torch.topk(adj_matrix, n_neighbor, dim=2, largest=False)
    nn_idx = nn_idx[:, :n_point, :]
    batch_size = x.size()[0]
    num_dim = x.size()[1]
    num_point = x.size()[2]
    point_expand = x[:, :, :n_point, :].expand(-1, -1, -1, num_point)
    nn_idx_expand = nn_idx.unsqueeze(1).expand(batch_size, num_dim, n_point, n_neighbor)
    pc_n = torch.gather(point_expand, -1, nn_idx_expand)
    return pc_n


def get_edge_feature(x, n_neighbor):
    if len(x.size()) == 3:
        x = x.unsqueeze(3)
    adj_matrix = pairwise_distance(x)
    _, nn_idx = torch.topk(adj_matrix, n_neighbor, dim=2, largest=False)
    point_cloud_neighbors = gather_neighbor(x, nn_idx, n_neighbor)
    point_cloud_center = x.expand(-1, -1, -1, n_neighbor)
    edge_feature = torch.cat((point_cloud_center, point_cloud_neighbors - point_cloud_center), dim=1)
    return edge_feature


# Part of the code is referred from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


def knn(x, k):
    """get k nearest neighbors based on distance in feature space

    Args:
        x: [b,dims(=3),num]
        k: number of neighbors to select

    Returns:
        k nearest neighbors (batch_size, num_points, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)  # [b,num,num]

    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # [b,1,num] #x ** 2
    # 2x1x2+2y1y2+2z1z2-x1^2-y1^2-z1^2-x2^2-y2^2-z2^2=-[(x1-x2)^2+(y1-y2)^2+(z1-z2)^2]
    pairwise_distance = -xx - inner
    pairwise_distance = pairwise_distance - xx.transpose(2, 1).contiguous()  # [b,num,num]
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
    dis, idx = pairwise_distance.topk(k=k + 1, dim=-1)  # (batch_size, num_points, k)
    dis = dis[:, :, 1:]
    idx = idx[:, :, 1:]
    return idx


def get_graph_feature(x, k=20, idx=None):
    # input x [B,dims,num]
    # output [B, dims*2, num, k] 邻域特征tensor
    """

    Args:
        x: [B,dims,num]
        k:
        idx:

    Returns:
        tensor [B, dims*2, num, k]
    """
    batch_size, dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1,
                                                               1) * num_points  # (batch_size, 1, 1) [0 num_points ... num_points*(B-1)]

    idx = idx + idx_base  # (batch_size, num_points, k)
    idx = idx.view(-1)  # (batch_size * num_points * k)
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size * num_points * k,dims)
    feature = feature.view(batch_size, num_points, k, dims)  # (batch_size, num_points, k, dims)
    x = x.view(batch_size, num_points, 1, dims).repeat(1, 1, k, 1)  # [B, num, k, dims]
    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)  # [B, dims*2, num, k]

    return feature


# input x [B,dims,num]
# output [B, dims*2, num, k]
def get_graph_featureNew(x, k=20, idx=None):
    batch_size, dims, num_points = x.size()
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    idx = idx.view(batch_size, num_points * k).unsqueeze(1).repeat(1, dims, 1)
    feature = torch.gather(x, index=idx, dim=2).view(batch_size, dims, num_points, k)
    x = x.unsqueeze(3).repeat(1, 1, 1, k)
    feature = torch.cat((feature, x), dim=1)  # [B, dims*2, num, k]

    return feature


def get_graph_feature_Origin(x, k=20, idx=None, cat=True):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x.detach(), k=k)  # (batch_size, num_points, k)

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1,
                                                               1) * num_points  # (batch_size, 1, 1) [0 num_points ... num_points*(B-1)]

    idx = idx + idx_base  # (batch_size, num_points, k)

    idx = idx.view(-1)  # (batch_size * num_points * k)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size * num_points * k,num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)  # (batch_size, num_points, k, num_dims)
    if cat:
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # [B, num, k, num_dims]
        feature = torch.cat((x, feature - x), dim=3).permute(0, 3, 1, 2)  # [B, num_dims*2, num, k]
    else:
        feature = feature.permute(0, 3, 1, 2)
    return feature
