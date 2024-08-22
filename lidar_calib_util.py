import open3d as o3d
import os
import numpy as np


class LidarCalibUtil:

    def __init__(self, args):
        self.args = args

    def store_point_cloud(self, point_cloud, file_name):
        """
        Store point cloud to a file.

        :param point_cloud: Point cloud data (numpy array, Nx3)
        :param file_name: File name to store point cloud
        """
        point_cloud = point_cloud.reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.io.write_point_cloud(
            os.path.join(self.args.folder_path, f"{file_name}.ply"), pcd
        )

    def depth2points(self, depth_map, focal_length, cx, cy):
        u = np.arange(depth_map.shape[1])
        v = np.arange(depth_map.shape[0])
        u, v = np.meshgrid(u, v)

        x = (u - cx) * depth_map / focal_length
        y = (v - cy) * depth_map / focal_length
        z = depth_map
        points = np.stack([x, y, z], axis=-1)
        points = points[:, 100:, :].reshape(-1, 3)
        points = points[:, ~np.isnan(points).any(axis=0)]
        points = points[:, ~np.isinf(points).any(axis=0)]

        return points
