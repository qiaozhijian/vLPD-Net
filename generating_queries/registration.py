import copy
import sys

import numpy as np
import open3d as o3d

# monkey patches visualization and provides helpers to load geometries
sys.path.append('..')


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    # a=np.asarray(source_temp.points)
    source_temp.transform(transformation)
    # b=np.asarray(source_temp.points)
    # o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size, is_voxel=True):
    if is_voxel:
        # print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)
    else:
        pcd_down = pcd
    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud("./test_data/cloud_bin_0.pcd")
    target = o3d.io.read_point_cloud("./test_data/cloud_bin_1.pcd")
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def refine_registration(source, target, init_transformation, voxel_size=-1):
    if voxel_size == -1:
        distance_threshold = 0.01
    else:
        distance_threshold = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
          % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def demo_open3d():
    voxel_size = 0.05  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    draw_registration_result(source_down, target_down, result_ransac.transformation)

    result_fast = execute_fast_global_registration(source_down, target_down,
                                                   source_fpfh, target_fpfh,
                                                   voxel_size)
    draw_registration_result(source_down, target_down, result_fast.transformation)

    result_icp = refine_registration(source, target, result_fast.transformation, voxel_size)
    draw_registration_result(source, target, result_icp.transformation)


def load_bin(path1, path2, voxel_size):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(np.fromfile(path1).reshape([-1, 3]))
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.fromfile(path2).reshape([-1, 3]))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size, is_voxel=False)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size, is_voxel=False)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def registration_numpy(pcl1, pcl2, trans_pre=np.eye(4)):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(pcl1)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(pcl2)

    draw_registration_result(source, target, np.eye(4))
    draw_registration_result(source, target, trans_pre)


def registration_withinit(path1, path2, trans_pre=[0, 0, 0]):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(np.fromfile(path1).reshape([-1, 3]))
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.fromfile(path2).reshape([-1, 3]))

    trans_init = np.asarray([[1.0, 0.0, 0.0, trans_pre[0]], [0.0, 1.0, 0.0, trans_pre[1]],
                             [0.0, 0.0, 1.0, trans_pre[2]], [0.0, 0.0, 0.0, 1.0]])

    result_icp = refine_registration(source, target, trans_init)
    # draw_registration_result(source, target, result_icp.transformation)

    T = np.asarray(result_icp.transformation)

    return T


if __name__ == '__main__':
    # demo_open3d()
    path1 = '../benchmark_datasets/GTA5/round1/pcl/1.bin'
    path2 = '../benchmark_datasets/GTA5/round1/pcl/2.bin'

    # path1 = '../benchmark_datasets/oxford/2014-05-19-13-20-57/pointcloud_20m_10overlap/1400505893170765.bin'
    # path2 = '../benchmark_datasets/oxford/2014-05-19-13-20-57/pointcloud_20m_10overlap/1400505894395159.bin'
    registration_withinit(path1, path2, voxel_size=0.00125)
