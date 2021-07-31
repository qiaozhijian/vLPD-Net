# Author: Zhijian Qiao
# Shanghai Jiao Tong University
# Code adapted from PointNetVlad code: https://github.com/jac99/MinkLoc3D.git

import glob
import math
import os
import pickle
import random

import numpy as np
import torch
import tqdm
from bitarray import bitarray
from scipy.linalg import expm, norm
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

from misc.log import log_string
from misc.utils import MinkLocParams

# import open3d as o3d

DEBUG = False


class OxfordDataset(Dataset):
    """
    Dataset wrapper for Oxford laser scans dataset from PointNetVLAD project.
    """

    def __init__(self, params, query_filepath, transform=None, set_transform=None, max_elems=None):
        # transform: transform applied to each element
        # set transform: transform applied to the entire set (anchor+positives+negatives); the same transform is applied
        assert os.path.exists(params.dataset_folder), 'Cannot access dataset path: {}'.format(params.dataset_folder)
        self.dataset_folder = params.dataset_folder
        self.query_filepath = os.path.join(params.queries_folder, query_filepath)
        self.da_filepath = os.path.join(params.queries_folder, params.da_train_file) if params.domain_adapt else None
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform
        self.max_elems = max_elems
        self.n_points = 4096  # pointclouds in the dataset are downsampled to 4096 points
        da_name = os.path.splitext(str(self.da_filepath))[0].split('/')[-1]
        cached_query_filepath = os.path.splitext(self.query_filepath)[0] + '_' + da_name + '_cached.pickle'

        if not os.path.exists(cached_query_filepath):
            # Pre-process query file
            self.queries = self.preprocess_queries(self.query_filepath, cached_query_filepath,
                                                   da_query_filepath=self.da_filepath)
        else:
            log_string('Loading preprocessed query file: {}...'.format(cached_query_filepath))
            with open(cached_query_filepath, 'rb') as handle:
                # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
                self.queries = pickle.load(handle)

        if max_elems is not None:
            filtered_queries = {}
            for ndx in self.queries:
                if ndx >= self.max_elems:
                    break
                filtered_queries[ndx] = {'query': self.queries[ndx]['query'],
                                         'positives': self.queries[ndx]['positives'][0:max_elems],
                                         'negatives': self.queries[ndx]['negatives'][0:max_elems],
                                         'da_query': self.queries[ndx]['da_query']}
            self.queries = filtered_queries

        log_string('{} queries in the dataset'.format(len(self)))

    def preprocess_queries(self, query_filepath, cached_query_filepath, da_query_filepath=None):
        log_string('Loading query file: {}...'.format(query_filepath))
        with open(query_filepath, 'rb') as handle:
            # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
            queries = pickle.load(handle)

        # Convert to bitarray
        # 每个问询的positive不再由所有positive的序号组成，而是大小与总样本数量相同的二进制向量存储（为1则是正样本）
        for ndx in tqdm.tqdm(queries):
            queries[ndx]['positives'] = set(queries[ndx]['positives'])
            queries[ndx]['negatives'] = set(queries[ndx]['negatives'])
            pos_mask = [e_ndx in queries[ndx]['positives'] for e_ndx in range(len(queries))]
            neg_mask = [e_ndx in queries[ndx]['negatives'] for e_ndx in range(len(queries))]
            queries[ndx]['positives'] = bitarray(pos_mask)
            queries[ndx]['negatives'] = bitarray(neg_mask)
            queries[ndx]['da_query'] = None

        if not da_query_filepath == None:
            with open(da_query_filepath, 'rb') as handle:
                da_queries = pickle.load(handle)
            index_da = np.random.choice(len(da_queries), len(queries), replace=len(queries) > len(da_queries))
            for ndx in range(len(queries)):
                queries[ndx]['da_query'] = da_queries[index_da[ndx]]['query']

        with open(cached_query_filepath, 'wb') as handle:
            pickle.dump(queries, handle)

        return queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        filename = self.queries[ndx]['query']
        query_pc = self.load_pc(filename)
        da_query_pc = None if self.queries[ndx]['da_query'] is None else self.load_pc(self.queries[ndx]['da_query'])
        query_source = query_pc
        R = np.eye(3)
        t = np.zeros(shape=(1, 3))
        if self.transform is not None:
            query_pc, R, t = self.transform(query_pc)
            da_query_pc = None if da_query_pc is None else self.transform(da_query_pc)
        return query_pc, ndx, da_query_pc, R, t, query_source

    def get_item_by_filename(self, filename):
        # Load point cloud and apply transform
        query_pc = self.load_pc(filename)
        if self.transform is not None:
            query_pc = self.transform(query_pc)
        return query_pc

    def get_items(self, ndx_l):
        # Load multiple point clouds and stack into (batch_size, n_points, 3) tensor
        clouds = [self[ndx][0] for ndx in ndx_l]
        clouds = torch.stack(clouds, dim=0)
        return clouds

    def get_set(self, anchor_ndx, positives_ndx, negatives_ndx):
        # Prepare a training set consisting of an anchor, positives and negatives
        # Each element is transformed using an item transform, then set transform is applied on all elements
        # Returns a (1+num_pos+num_neg, n_points, 3) tensor or None for broken clouds
        anchor, _ = self[anchor_ndx]
        if len(anchor) == 0:
            return None, None

        anchor = anchor.unsqueeze(dim=0)  # Make it (1, num_points, 3) tensor
        positives = self.get_items(positives_ndx)
        negatives = self.get_items(negatives_ndx)
        set = torch.cat([anchor, positives, negatives], dim=0)

        if self.set_transform is not None:
            set = self.set_transform(set)

        set_ndx = [anchor_ndx] + positives_ndx + negatives_ndx  # Indexes of elements in the set
        return set, set_ndx

    def get_positives_ndx(self, ndx):
        # Get list of indexes of similar clouds
        return self.queries[ndx]['positives'].search(bitarray([True]))

    def get_negatives_ndx(self, ndx):
        # Get list of indexes of dissimilar clouds
        return self.queries[ndx]['negatives'].search(bitarray([True]))

    def load_pc(self, filename):
        # Load point cloud, does not apply any transform
        # Returns Nx3 matrix
        file_path = os.path.join(self.dataset_folder, filename)
        pc = np.fromfile(file_path, dtype=np.float64)
        # coords are within -1..1 range in each dimension
        assert pc.shape[0] == self.n_points * 3, "Error in point cloud shape: {}".format(filename)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        pc = torch.tensor(pc, dtype=torch.float)
        return pc

    def get_positives_mask(self, labels):
        # Compute n_elem x n_elem boolean mask for positive examples
        return [True for e in labels]

    def get_negatives_mask(self, labels):
        # Compute n_elem x n_elem boolean mask for negative examples
        return [True for e in labels]


class TrainTransform:
    def __init__(self, aug_mode):
        # 1 is default mode, no transform
        self.aug_mode = aug_mode
        if self.aug_mode == 0:
            self.transform = []
        elif self.aug_mode >= 1:
            self.transform = []
            self.transform.append(RandomRotation(max_theta=15, max_theta2=None, axis=np.array([0, 0, 1])))
            self.transform.append(RandomRotation(max_theta=5, max_theta2=None, axis=np.array([0, 1, 0])))
            self.transform.append(RandomRotation(max_theta=5, max_theta2=None, axis=np.array([1, 0, 0])))
            self.transform.append(RandomTranslation(max_delta=0.05))
            self.transform.append(JitterPoints(sigma=0.001, clip=0.002))
            self.transform.append(RemoveRandomPoints(r=(0.0, 0.1)))
            # self.transform.append(RemoveRandomBlock(p=0.4))
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))

    def __call__(self, e):
        n = len(self.transform)
        R_a = np.eye(3)
        t_a = np.zeros((3, 1))
        for i in range(n):
            cur_tf = self.transform[i]
            if type(cur_tf).__name__ == 'RandomRotation':
                e, R_i = cur_tf(e)
                R_a = R_i @ R_a
                t_a = R_i @ t_a
            elif type(cur_tf).__name__ == 'RandomTranslation':
                e, t_i = cur_tf(e)
                R_a = R_a
                t_a = t_a + t_i.T
            else:
                e = cur_tf(e)
        return e, R_a, t_a.T


class TrainSetTransform:
    def __init__(self, aug_mode):
        # 1 is default mode, no transform
        self.aug_mode = aug_mode
        self.transform = []
        if self.aug_mode < 2:
            self.transform = []
        elif self.aug_mode == 2:
            self.transform.append(RandomRotation(max_theta=5, max_theta2=None, axis=np.array([0, 0, 1])))
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))

    def __call__(self, e):
        n = len(self.transform)
        R_a = np.eye(3)
        t_a = np.zeros((3, 1))
        for i in range(n):
            cur_tf = self.transform[i]
            if type(cur_tf).__name__ == 'RandomRotation':
                e, R_i = cur_tf(e)
                R_a = R_i @ R_a
                t_a = R_i @ t_a
            elif type(cur_tf).__name__ == 'RandomTranslation':
                e, t_i = cur_tf(e)
                R_a = R_a
                t_a = t_a + t_i.T
            else:
                e = cur_tf(e)
        return e, R_a, t_a.T


class RandomFlip:
    def __init__(self, p):
        # p = [p_x, p_y, p_z] probability of flipping each axis
        assert len(p) == 3
        assert 0 <= sum(p) <= 1, 'sum(p) must be in (0, 1] range, is: {}'.format(sum(p))
        self.p = p
        self.p_cum_sum = np.cumsum(p)

    def __call__(self, coords):
        r = random.random()
        if r <= self.p_cum_sum[0]:
            # Flip the first axis
            coords[..., 0] = -coords[..., 0]
        elif r <= self.p_cum_sum[1]:
            # Flip the second axis
            coords[..., 1] = -coords[..., 1]
        elif r <= self.p_cum_sum[2]:
            # Flip the third axis
            coords[..., 2] = -coords[..., 2]

        return coords


class RandomRotation:
    def __init__(self, axis=None, max_theta=180, max_theta2=15):
        self.axis = axis
        self.max_theta = max_theta  # Rotation around axis
        self.max_theta2 = max_theta2  # Smaller rotation in random direction

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta)).astype(np.float32)

    def __call__(self, coords):

        if self.axis is not None:
            axis = self.axis
        else:
            axis = np.random.rand(3) - 0.5
            self.axis = axis
        #     正负 max_theta
        self.last_theta = (np.pi * self.max_theta / 180) * 2 * (np.random.rand(1) - 0.5)
        R = self._M(axis, self.last_theta)
        if self.max_theta2 is None:
            coords = coords @ R
            R_a = R.T
        else:
            # 旋转轴随机
            self.last_theta2 = (np.pi * self.max_theta2 / 180) * 2 * (np.random.rand(1) - 0.5)
            self.axis2 = np.random.rand(3) - 0.5
            R_n = self._M(self.axis2, self.last_theta2)
            coords = coords @ R @ R_n
            R_a = R_n.T @ R.T

        return coords, R_a


class RandomTranslation:
    def __init__(self, max_delta=0.05):
        self.max_delta = max_delta

    def __call__(self, coords):
        trans = self.max_delta * np.random.randn(1, 3)
        return coords + trans.astype(np.float32), trans.astype(np.float32)


class RandomScale:
    def __init__(self, min, max):
        self.scale = max - min
        self.bias = min

    def __call__(self, coords):
        s = self.scale * np.random.rand(1) + self.bias
        return coords * s.astype(np.float32)


class RandomShear:
    def __init__(self, delta=0.1):
        self.delta = delta

    def __call__(self, coords):
        T = np.eye(3) + self.delta * np.random.randn(3, 3)
        return coords @ T.astype(np.float32)


class JitterPoints:
    def __init__(self, sigma=0.01, clip=None, p=1.):
        assert 0 < p <= 1.
        assert sigma > 0.

        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, e):
        """ Randomly jitter points. jittering is per point.
            Input:
              BxNx3 array, original batch of point clouds
            Return:
              BxNx3 array, jittered batch of point clouds
        """

        sample_shape = (e.shape[0],)
        if self.p < 1.:
            # Create a mask for points to jitter
            m = torch.distributions.categorical.Categorical(probs=torch.tensor([1 - self.p, self.p]))
            mask = m.sample(sample_shape=sample_shape)
        else:
            mask = torch.ones(sample_shape, dtype=torch.int64)

        mask = mask == 1
        jitter = self.sigma * torch.randn_like(e[mask])

        if self.clip is not None:
            jitter = torch.clamp(jitter, min=-self.clip, max=self.clip)

        e[mask] = e[mask] + jitter
        return e


class RemoveRandomPoints:
    def __init__(self, r):
        if type(r) is list or type(r) is tuple:
            assert len(r) == 2
            assert 0 <= r[0] <= 1
            assert 0 <= r[1] <= 1
            self.r_min = float(r[0])
            self.r_max = float(r[1])
        else:
            assert 0 <= r <= 1
            self.r_min = None
            self.r_max = float(r)

    def __call__(self, e):
        n = len(e)
        if self.r_min is None:
            r = self.r_max
        else:
            # Randomly select removal ratio
            r = random.uniform(self.r_min, self.r_max)

        mask = np.random.choice(range(n), size=int(n * r), replace=False)  # select elements to remove
        e[mask] = torch.zeros_like(e[mask])
        return e


class RemoveRandomBlock:
    """
    Randomly remove part of the point cloud. Similar to PyTorch RandomErasing but operating on 3D point clouds.
    Erases fronto-parallel cuboid.
    Instead of erasing we set coords of removed points to (0, 0, 0) to retain the same number of points
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def get_params(self, coords):
        # Find point cloud 3D bounding box
        flattened_coords = coords.view(-1, 3)
        min_coords, _ = torch.min(flattened_coords, dim=0)
        max_coords, _ = torch.max(flattened_coords, dim=0)
        span = max_coords - min_coords
        area = span[0] * span[1]
        erase_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        h = math.sqrt(erase_area * aspect_ratio)
        w = math.sqrt(erase_area / aspect_ratio)

        x = min_coords[0] + random.uniform(0, 1) * (span[0] - w)
        y = min_coords[1] + random.uniform(0, 1) * (span[1] - h)

        return x, y, w, h

    def __call__(self, coords):
        if random.random() < self.p:
            x, y, w, h = self.get_params(coords)  # Fronto-parallel cuboid to remove
            mask = (x < coords[..., 0]) & (coords[..., 0] < x + w) & (y < coords[..., 1]) & (coords[..., 1] < y + h)
            coords[mask] = torch.zeros_like(coords[mask])
        return coords


def load_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '../../dataset/benchmark_datasets/oxford')
    all_data = []
    if partition == 'train':
        for seq in glob.glob(os.path.join(DATA_DIR, '*')):
            for idx, filename in enumerate(glob.glob('{}/pointcloud_20m_10overlap/*.bin'.format(seq))):
                if not idx % 4 == 0:
                    # if idx % 10==0:
                    pc = np.fromfile(filename, dtype=np.float64)
                    pc = np.reshape(pc, (1, pc.shape[0] // 3, 3))
                    all_data.append(pc)
    else:
        for seq in glob.glob(os.path.join(DATA_DIR, '*')):
            for idx, filename in enumerate(glob.glob('{}/pointcloud_20m_10overlap/*.bin'.format(seq))):
                if idx % 4 == 0:
                    pc = np.fromfile(filename, dtype=np.float64)
                    pc = np.reshape(pc, (1, pc.shape[0] // 3, 3))
                    all_data.append(pc)
    all_data = np.concatenate(all_data, axis=0)
    return all_data


class Oxford(Dataset):
    def __init__(self, params: MinkLocParams, partition='train'):
        self.num_points = params.reg.num_points
        self.overlap = params.reg.overlap
        self.partition = partition
        print('Load Oxford Dataset')
        self.data = load_data(partition)
        self.partial = True

    def __getitem__(self, item):

        # [num,num_dim]
        pointcloud = self.data[item]

        if self.partition != 'train':
            np.random.seed(item)

        # anglex = (np.random.uniform() - 0.5) * 2 * 5.0 / 180.0 * np.pi
        # angley = (np.random.uniform() - 0.5) * 2 * 5.0 / 180.0 * np.pi
        # anglez = (np.random.uniform() - 0.5) * 2 * 35.0 / 180.0 * np.pi

        anglex = (np.random.uniform()) * 5.0 / 180.0 * np.pi
        angley = (np.random.uniform()) * 5.0 / 180.0 * np.pi
        anglez = (np.random.uniform()) * 35.0 / 180.0 * np.pi

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T

        if self.partition == 'train':
            translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                       np.random.uniform(-0.5, 0.5)])
        else:
            translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                       np.random.uniform(-0.5, 0.5)])

        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = ((np.random.permutation(pointcloud))[: self.num_points]).T
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])

        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        # if self.overlap < 1.0:
        #     pointcloud1 = self.nearest_neighbor(pointcloud1, self.overlap)
        pointcloud2 = np.random.permutation(pointcloud2.T).T
        # if self.overlap < 1.0:
        #     pointcloud2 = self.nearest_neighbor(pointcloud2, self.overlap)

        # [3,num_points] (3,)
        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32'), 0

    def __len__(self):
        return self.data.shape[0]
        # return 100

    def nearest_neighbor(self, dst, reserve):
        dst = dst.T
        num = np.max([dst.shape[0], dst.shape[1]])
        num = int(num * reserve)
        src = dst[-1, :].reshape(1, -1)
        neigh = NearestNeighbors(n_neighbors=num)
        neigh.fit(dst)
        indices = neigh.kneighbors(src, return_distance=False)
        indices = indices.ravel()
        return dst[indices, :].T
