import numpy as np
import os
from camera_lidar_calibration import CameraLidarCalibration
from typing import Optional, Callable
import cv2
import matplotlib.pyplot as plt


class DatasetProcess:

    def __init__(self, folder_root: str):
        self.folder_root = folder_root

    def search_folder(self, folder: str, tree_depth=0):
        subs = os.listdir(folder)
        data_list: list[str] = []
        for sub in subs:
            sub_folder = os.path.join(folder, sub)
            if not os.path.isdir(sub_folder):
                continue
            if os.path.exists(os.path.join(sub_folder, "raw.npz")) and os.path.exists(
                os.path.join(sub_folder, "post.npz")
            ):
                data_list.append(sub_folder)
                continue
            if tree_depth < 3:
                data_list += self.search_folder(sub_folder, tree_depth=tree_depth + 1)
        data_list.sort()
        return data_list

    def scene_get_lidar_points(self, scene: str):
        raw = np.load(os.path.join(scene, "raw.npz"))
        if "points" in raw:
            return raw["points"]
        return self.lidar_points_2d_to_3d(raw["ranges"])

    def lidar_points_2d_to_3d(self, ranges: np.ndarray):
        WIDTH = 1024
        HEIGHT = 128
        V_FOV = np.pi / 8
        V_FOV_MIN = -V_FOV / 2
        V_FOV_MAX = V_FOV / 2

        phi = np.linspace(V_FOV_MIN, V_FOV_MAX, HEIGHT)
        theta = np.linspace(0, 2 * np.pi, WIDTH, endpoint=False)

        x = ranges * np.cos(phi[:, np.newaxis]) * np.cos(theta)
        y = ranges * np.cos(phi[:, np.newaxis]) * np.sin(theta)
        z = ranges * np.sin(phi[:, np.newaxis])

        return np.stack([x, y, z], axis=-1)

    def process_scene_list(
        self,
        scene_list: list[str],
        project_mtx: np.ndarray,
        log_callback: Callable[[int, str], None] = lambda x, y: print(f"{x}%: {y}"),
    ):
        for idx, scene in enumerate(scene_list):
            p_idx = idx * 100 // len(scene_list)
            log_callback(p_idx, f"Processing {scene}")
            output_dict, plot_dict = self.process_scene(scene, project_mtx)
            depth_median = max(
                np.median(output_dict["depth"]),
                np.median(output_dict["projected_depth_rendered"]),
            )
            self.plot_outputs(
                scene,
                plot_dict,
                {
                    "disparity": 100,
                    "depth": max(10000, depth_median * 5),
                    "projected_depth_rendered": max(10000, depth_median * 5),
                },
            )
            self.update_post_npz(output_dict, scene)
            log_callback(p_idx, f"Processed {scene} of {idx}/{len(scene_list)}")

    def update_post_npz(self, update_dict: dict[str, np.ndarray], scene: str):
        post = np.load(os.path.join(scene, "post.npz"))
        updated_post = {**post}
        for key, value in update_dict.items():
            updated_post[key] = value
        np.savez(os.path.join(scene, "post.npz"), **updated_post)

    def process_scene(self, scene: str, project_mtx: np.ndarray):
        raw = np.load(os.path.join(scene, "raw.npz"))
        post = np.load(os.path.join(scene, "post.npz"))
        k_left = post["k_left"]
        baseline = np.linalg.norm(post["T"])
        disparity = post["disparity"]
        depth = post["depth"]
        image_left = cv2.imread(os.path.join(scene, "left_rectified.png"))
        image_right = cv2.imread(os.path.join(scene, "right_rectified.png"))
        points = self.scene_get_lidar_points(scene)
        remap_mask: Optional[np.ndarray] = (
            self.remap_mask if hasattr(self, "remap_mask") else None
        )
        calib = CameraLidarCalibration(
            args={},
            disparity_map=disparity,
            lidar_point_map=points,
            baseline_distance=baseline,
            K=k_left,
            init_transform=project_mtx,
            remap_mask=remap_mask,
        )
        h = disparity.shape[0]
        w = disparity.shape[1]
        if remap_mask is None:

            R = post["R"]
            T = post["T"]
            d_left = post["d_left"]
            k_right = post["k_right"]
            d_right = post["d_right"]
            self.remap_mask = calib.compute_remap_mask(
                h, w, k_left, d_left, k_right, d_right, R, T
            )

        disparity[self.remap_mask == 0] = 0
        depth[self.remap_mask == 0] = 0
        projected_depth = calib.transform_lidarpoints_to_image(points, depth)
        projected_depth_rendered = calib.render_2dpoint_to_image(
            projected_depth, h, w, use_color=False
        )

        return {
            "projected_depth": projected_depth,
            "disparity": disparity,
            "depth": depth,
            "projected_depth_rendered": projected_depth_rendered,
        }, {
            "image_left": image_left,
            "image_right": image_right,
            "disparity": disparity,
            "depth": depth,
            "projected_depth_rendered": projected_depth_rendered,
        }

    def plot_outputs(
        self,
        scene_path: str,
        output_dict: dict[str, np.ndarray],
        value_range_max: dict[str, float] = {},
    ):
        fig, axs = plt.subplots(1, len(output_dict), figsize=(8 * len(output_dict), 6))

        for i, (key, value) in enumerate(output_dict.items()):
            if len(value.shape) == 2:
                value[value < 0] = 0
                vmax = value.max()
                if key in value_range_max:
                    value[value > value_range_max[key]] = value_range_max[key]
                    vmax = value_range_max[key]
                im = axs[i].imshow(value, cmap="magma", vmax=vmax)

                fig.colorbar(im, ax=axs[i])
            else:
                axs[i].imshow(value)
            axs[i].set_title(key)
        plot_filename = os.path.join(scene_path, "lidar_plot.png")
        plt.savefig(plot_filename)
        plt.close()
