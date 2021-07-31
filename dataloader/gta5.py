import os
import pickle

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

from misc.utils import MinkLocParams


def load_data(params, partition):
    query_filepath = os.path.join(params.queries_folder, params.train_file)
    with open(query_filepath, 'rb') as handle:
        # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
        queries = pickle.load(handle)
    num = len(queries)
    all_data = queries
    # if partition=='train':
    #     for idx in range(num):
    #         if idx % 4!=0:
    #             all_data.append(queries[idx])
    # else:
    #     for idx in range(num):
    #         if idx % 4==0:
    #             all_data.append(queries[idx])
    return all_data


class GTAV5(Dataset):
    def __init__(self, params: MinkLocParams, partition='train'):
        self.num_points = params.reg.num_points
        self.overlap = params.reg.overlap
        self.partition = partition
        print('Load GTAV5 Dataset')
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(BASE_DIR, '../../dataset/benchmark_datasets/')
        self.data = load_data(params, partition)
        # self.partial = True

    def __getitem__(self, item):

        if self.partition != 'train':
            np.random.seed(item)

        # [num,num_dim]
        query_dict = self.data[item]

        filename = os.path.join(self.DATA_DIR, query_dict['query'])
        pointcloud = np.fromfile(filename, dtype=np.float64)
        pointcloud = np.reshape(pointcloud, (pointcloud.shape[0] // 3, 3))
        pointcloud1 = (np.random.permutation(pointcloud)[: self.num_points]).T

        num_samples = len(query_dict['positives'])
        pos_idx = np.random.randint(num_samples)
        positive_dict = self.data[query_dict['positives'][pos_idx]]
        filename = os.path.join(self.DATA_DIR, positive_dict['query'])
        pointcloud = np.fromfile(filename, dtype=np.float64)
        pointcloud = np.reshape(pointcloud, (pointcloud.shape[0] // 3, 3))
        pointcloud2 = (np.random.permutation(pointcloud)[: self.num_points]).T

        positive_T = query_dict['positives_T'][pos_idx]

        rotation_ab = positive_T[:3, :3]
        translation_ab = positive_T[:3, 3].reshape(3, 1)
        pointcloud1 = rotation_ab @ pointcloud1 + translation_ab

        # self.visual_pcl_simple(pcl1=pointcloud1.T, pcl2=pointcloud2.T)

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

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud2.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])

        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        # [3,num_points] (3,)
        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32'), 0

    def __len__(self):
        return len(self.data)
        # return 100

    def visual_pcl_simple(self, pcl1, pcl2, name='Open3D Origin'):
        # pcl: N,3
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pcl1[:, :3])
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pcl2[:, :3])
        pcd1.paint_uniform_color([1, 0.706, 0])
        pcd2.paint_uniform_color([0, 0.651, 0.929])
        o3d.visualization.draw_geometries([pcd1, pcd2], window_name=name, width=1920, height=1080,
                                          left=50,
                                          top=50,
                                          point_show_normal=False, mesh_show_wireframe=False,
                                          mesh_show_back_face=False)

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
