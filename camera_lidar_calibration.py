import numpy as np
from typing import List, Tuple
import argparse

import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from typing import Optional
import os

MAX_DEPTH = 5000


class CameraLidarCalibration:
    def __init__(
        self,
        args,
        disparity_map: np.ndarray,
        lidar_point_map: np.ndarray,
        K: np.ndarray,
        baseline_distance: float,
        uv_pairs: List[Tuple[int, int, int, int]],
    ):
        """
        CameraLidarCalibration 클래스를 초기화합니다.

        :param disparity_map: hxw 크기의 카메라 disparity map (numpy 배열, float형)
        :param lidar_point_map: 1024x128x3 크기의 라이다 3D 포인트 좌표 map (numpy 배열)
        :param focal_length: 카메라의 초점 거리 (focal length)
        :param baseline_distance: 스테레오 카메라의 베이스라인 거리
        :param uv_pairs: (u, v) 좌표 페어 리스트 [(u_cam, v_cam, u_lidar, v_lidar), ...]
        """
        self.args = args
        self.disparity_map = disparity_map
        self.lidar_point_map = lidar_point_map
        self.focal_length = K[0, 0]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        self.baseline_distance = baseline_distance
        self.uv_pairs = uv_pairs

        self.colorbar_magma = cv2.applyColorMap(
            np.arange(256).astype(np.uint8), cv2.COLORMAP_MAGMA
        )

    def disparity_to_depth(self, disparity):
        """
        Disparity 값을 깊이 값으로 변환합니다.

        :param disparity: Disparity 값
        :return: 깊이 값 (Depth)
        """

        return self.focal_length * self.baseline_distance / disparity

    def compute_camera_points(self):
        """
        카메라 이미지에서 uv 좌표로부터 3D 포인트를 계산합니다.

        :return: 카메라 좌표계의 3D 포인트 리스트
        """
        camera_points = []
        for u_cam, v_cam, _, _ in self.uv_pairs:

            disparity = self.disparity_map[v_cam, u_cam]
            depth = self.disparity_to_depth(disparity)

            x = (u_cam - self.cx) * depth / self.focal_length
            y = (v_cam - self.cy) * depth / self.focal_length
            z = depth

            camera_points.append([x, y, z])
        return np.array(camera_points)

    def get_lidar_points(self):
        """
        라이다 이미지에서 uv 좌표로부터 3D 포인트를 가져옵니다.

        :return: 라이다 좌표계의 3D 포인트 리스트
        """
        lidar_points = []
        for _, _, u_lidar, v_lidar in self.uv_pairs:
            lidar_points.append(self.lidar_point_map[v_lidar, u_lidar] * 1000)
        return np.array(lidar_points)

    def estimate_transform(self):
        """
        카메라와 라이다 간의 변환 행렬을 추정합니다.

        :return: 4x4 변환 행렬
        """
        camera_points = self.compute_camera_points()
        lidar_points = self.get_lidar_points()

        centroid_camera = np.mean(camera_points, axis=0)
        centroid_lidar = np.mean(lidar_points, axis=0)

        camera_centered = camera_points - centroid_camera
        lidar_centered = lidar_points - centroid_lidar

        H = lidar_centered.T @ camera_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_camera.T - R @ centroid_lidar.T

        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = t

        points_lidar = self.lidar_point_map.reshape(-1, 3) * 1000

        transform_matrix = self.estimate_open3d_transform(
            self.computeCameraPointCloud(), points_lidar, transform_matrix
        )
        self.transform_matrix = transform_matrix
        return transform_matrix

    def computeCameraPointCloud(self):
        """
        Return pointcloud from disparity map stored in npz file
        """
        depthmap = self.disparity_to_depth(self.disparity_map)
        points_camera = self.depth2points(depthmap)
        return points_camera

    def depth2points(self, depth_map):
        u = np.arange(depth_map.shape[1])
        v = np.arange(depth_map.shape[0])
        u, v = np.meshgrid(u, v)

        x = (u - self.cx) * depth_map / self.focal_length
        y = (v - self.cy) * depth_map / self.focal_length
        z = depth_map
        points = np.stack([x, y, z], axis=-1)
        points = points[:, 100:, :].reshape(-1, 3)
        points = points[:, ~np.isnan(points).any(axis=0)]
        points = points[:, ~np.isinf(points).any(axis=0)]

        return points

    def estimate_open3d_transform(
        self, points_camera, points_lidar, trans_init, threshold=100
    ):
        # 포인트 클라우드를 Open3D 포맷으로 변환

        pcd_camera = o3d.geometry.PointCloud()
        pcd_lidar = o3d.geometry.PointCloud()

        pcd_camera.points = o3d.utility.Vector3dVector(points_camera)
        pcd_lidar.points = o3d.utility.Vector3dVector(points_lidar)

        # ICP (Iterative Closest Point) 매칭 수행
        # 거리 임계값

        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd_lidar,
            pcd_camera,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        return reg_p2p.transformation

    def transform_lidarpoints_to_image(
        self, lidar_points: np.ndarray, image_shape: Tuple[int, int]
    ):

        height, width = image_shape
        transform_matrix = self.transform_matrix
        lidar_points = lidar_points.reshape(-1, 3) * 1000

        # 3D 라이다 포인트를 4xN 행렬로 변환

        lidar_points = np.concatenate(
            [lidar_points, np.ones((lidar_points.shape[0], 1))], axis=1
        ).T

        # 변환 행렬을 사용하여 라이다 포인트를 카메라 좌표계로 변환
        # camera_points = transform_matrix @ lidar_points
        camera_points = np.linalg.inv(transform_matrix) @ lidar_points

        # 카메라 좌표계의 3D 포인트를 2D 이미지 좌표로 변환
        u = camera_points[0] * self.focal_length / camera_points[2] + self.cx
        v = camera_points[1] * self.focal_length / camera_points[2] + self.cy
        u = u
        v = v
        depth = camera_points[2]

        camera_surface = np.stack([u, v, depth], axis=1)
        lidar_points = lidar_points.T
        lidar_points = lidar_points[lidar_points[:, 2] <= 0]
        plot_png = self.plot_points_cloud(
            [
                lidar_points,
                self.computeCameraPointCloud(),
                camera_points.T,
                camera_surface,
            ]
        )
        camera_surface_filtered = camera_surface[
            (camera_surface[:, 0] > 0)
            & (camera_surface[:, 0] < width)
            & (camera_surface[:, 1] > 0)
            & (camera_surface[:, 1] < height)
        ]

        canvas = np.zeros((height, width), dtype=np.uint8)

        for u, v, depth in camera_surface_filtered:
            depth_color = depth / MAX_DEPTH * 255
            depth_color = int(np.clip(depth_color, 0, 255))

            cv2.circle(canvas, (int(u), int(v)), 5, [depth_color], -1)

        plot_png = cv2.resize(
            plot_png, (int(width / height * plot_png.shape[1]), height)
        )

        canvas = cv2.applyColorMap(canvas, cv2.COLORMAP_MAGMA)
        return np.concatenate([plot_png, canvas], axis=1)

    def plot_points_cloud(
        self,
        points_array: List[np.ndarray],
    ):
        fig = plt.figure(figsize=(5 * len(points_array), 10))
        for i, points in enumerate(points_array):
            ax = fig.add_subplot(1, len(points_array), i + 1, projection="3d")

            filtered_points = points[
                (points[:, 0] > -MAX_DEPTH)
                & (points[:, 0] < MAX_DEPTH)
                & (points[:, 1] > -MAX_DEPTH)
                & (points[:, 1] < MAX_DEPTH)
                & (points[:, 2] > -MAX_DEPTH)
                & (points[:, 2] < MAX_DEPTH)
            ]
            ax.set_xlim(-MAX_DEPTH, MAX_DEPTH)
            ax.set_ylim(-MAX_DEPTH, MAX_DEPTH)
            ax.set_zlim(-MAX_DEPTH, MAX_DEPTH)

            color = (
                np.clip(np.linalg.norm(filtered_points, axis=-1) / MAX_DEPTH, 0, 1)
                * 255
            ).astype(np.uint8)
            color = self.colorbar_magma[color] / 255.0

            ax.scatter(
                filtered_points[:, 0],
                filtered_points[:, 1],
                filtered_points[:, 2],
                c=color,
            )

        plt.savefig("capture_image.png")
        plt.close()
        return cv2.imread("capture_image.png")


# 사용 예시
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True)
    args = parser.parse_args()
