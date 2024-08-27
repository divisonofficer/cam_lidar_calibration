import tkinter as tk
import math
import numpy as np
from typing import Callable


class TransformLabel:

    _compute_transform: Callable[[bool, bool], None]
    _save_transform: Callable[[np.ndarray], None]
    _read_transform: Callable[[], np.ndarray]

    def compute_transform(self):
        use_pair = self.tf_option_use_pair.get()
        use_open3d = self.tf_option_use_open3d.get()
        self._compute_transform(use_pair, use_open3d)

    def read_transform(self):
        mtx = self._read_transform()
        if mtx is None:
            return
        self.update_transform_result(mtx)

    def save_transform(self):
        mtx = self.get_quaternion_from_eular()
        if mtx is None:
            return
        self._save_transform(mtx)

    def __init__(self, root: tk.Misc):
        self.root = root
        self.frame = tk.Frame(self.root)

        self.eular_frame = tk.Frame(self.frame)
        self.eular_frame.pack(side=tk.LEFT)

        eular_rot_frame = tk.Frame(self.eular_frame)
        self.eular_x = tk.Text(eular_rot_frame, height=1, width=10)
        self.eular_x.pack(side=tk.LEFT)
        self.eular_y = tk.Text(eular_rot_frame, height=1, width=10)
        self.eular_y.pack(side=tk.LEFT)
        self.eular_z = tk.Text(eular_rot_frame, height=1, width=10)
        self.eular_z.pack(side=tk.LEFT)
        eular_t_frame = tk.Frame(self.eular_frame)
        self.eular_tx = tk.Text(eular_t_frame, height=1, width=10)
        self.eular_tx.pack(side=tk.LEFT)
        self.eular_ty = tk.Text(eular_t_frame, height=1, width=10)
        self.eular_ty.pack(side=tk.LEFT)
        self.eular_tz = tk.Text(eular_t_frame, height=1, width=10)
        self.eular_tz.pack(side=tk.LEFT)
        eular_rot_frame.pack(side=tk.TOP)
        eular_t_frame.pack(side=tk.TOP)
        """
        Action Button Frame
        """
        action_frame = tk.Frame(self.eular_frame)
        action_frame.pack(side=tk.TOP)

        transform_button = tk.Button(
            action_frame, text="Compute Transform", command=self.compute_transform
        )
        transform_button.pack(pady=5)

        self.tf_option_use_pair = tk.BooleanVar(value=False)
        self.tf_option_use_open3d = tk.BooleanVar(value=False)

        cb_use_pair = tk.Checkbutton(
            action_frame, text="Use Pair", variable=self.tf_option_use_pair
        )
        cb_use_pair.pack(side=tk.LEFT)
        cb_use_open3d = tk.Checkbutton(
            action_frame, text="Use Open3D", variable=self.tf_option_use_open3d
        )
        cb_use_open3d.pack(side=tk.LEFT)

        save_transform_button = tk.Button(
            action_frame, text="Save Transform", command=self.save_transform
        )
        save_transform_button.pack(side=tk.LEFT, pady=5)
        read_transform_button = tk.Button(
            action_frame, text="Read Transform", command=self.read_transform
        )
        read_transform_button.pack(side=tk.LEFT, pady=5)

        """
        Euler Input Frame
        """
        self.eular_to_q_btn = tk.Button(
            self.frame,
            text="Eular to Quaternion",
            command=self.update_quaternion_from_eular,
        )
        self.eular_to_q_btn.pack(side=tk.LEFT)

        self.transform_result_text = tk.Text(self.frame, height=10, width=40)
        self.transform_result_text.pack(side=tk.LEFT, fill=tk.BOTH)

        self.transform_result_image_view = tk.Label(self.frame)
        self.transform_result_image_view.pack(side=tk.RIGHT, fill=tk.BOTH)

        """
        Depth Option Frame
        """
        self.depth_option_frame = tk.Frame(self.frame)
        self.depth_option_frame.pack(side=tk.LEFT)
        depth_max_label = tk.Label(self.depth_option_frame, text="Depth Max")
        depth_max_label.pack(side=tk.LEFT)
        self.depth_max_input = tk.Text(self.depth_option_frame, height=1, width=10)
        self.depth_max_input.pack(side=tk.LEFT)

    def __call__(self):
        return self.frame

    def get_depth_max_option(self):
        try:
            return float(self.depth_max_input.get(1.0, tk.END))
        except ValueError:
            return 10000.0

    def update_transform_result(self, mtx: np.ndarray):
        self.transform_result_text.delete(1.0, tk.END)
        self.transform_result_text.insert(tk.END, f"{mtx}")

        for i, text in enumerate([self.eular_tx, self.eular_ty, self.eular_tz]):
            text.delete(1.0, tk.END)
            text.insert(tk.END, f"{mtx[i, 3]}")

        eular = self.rotation_matrix_to_euler_angles(mtx[:3, :3])
        for i, text in enumerate([self.eular_x, self.eular_y, self.eular_z]):
            text.delete(1.0, tk.END)
            text.insert(tk.END, f"{eular[i]}")

    def rotation_matrix_to_euler_angles(self, R):
        """
        Converts a 3x3 rotation matrix to Euler angles (x, y, z).

        :param R: 3x3 rotation matrix
        :return: A tuple of Euler angles (x, y, z)
        """
        assert R.shape == (3, 3)

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        x = round(math.degrees(x), 3)
        y = round(math.degrees(y), 3)
        z = round(math.degrees(z), 3)
        return x, y, z

    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        Converts Euler angles (roll, pitch, yaw) to a quaternion.

        :param roll: Rotation angle around the x-axis (in radians)
        :param pitch: Rotation angle around the y-axis (in radians)
        :param yaw: Rotation angle around the z-axis (in radians)
        :return: A quaternion represented as [qw, qx, qy, qz]
        """
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return np.array([qw, qx, qy, qz])

    def quaternion_to_rotation_matrix(self, q):
        """
        Converts a quaternion to a 3x3 rotation matrix.

        :param q: A quaternion represented as [qw, qx, qy, qz]
        :return: A 3x3 rotation matrix
        """
        qw, qx, qy, qz = q

        # Compute the matrix elements
        R = np.array(
            [
                [
                    1 - 2 * qy**2 - 2 * qz**2,
                    2 * qx * qy - 2 * qz * qw,
                    2 * qx * qz + 2 * qy * qw,
                ],
                [
                    2 * qx * qy + 2 * qz * qw,
                    1 - 2 * qx**2 - 2 * qz**2,
                    2 * qy * qz - 2 * qx * qw,
                ],
                [
                    2 * qx * qz - 2 * qy * qw,
                    2 * qy * qz + 2 * qx * qw,
                    1 - 2 * qx**2 - 2 * qy**2,
                ],
            ]
        )

        return R

    def get_quaternion_from_eular(self):
        try:
            x = float(self.eular_x.get(1.0, tk.END))
            y = float(self.eular_y.get(1.0, tk.END))
            z = float(self.eular_z.get(1.0, tk.END))

            tx = float(self.eular_tx.get(1.0, tk.END))
            ty = float(self.eular_ty.get(1.0, tk.END))
            tz = float(self.eular_tz.get(1.0, tk.END))
        except ValueError:
            return None
        x = math.radians(x)
        y = math.radians(y)
        z = math.radians(z)
        R = self.euler_to_quaternion(x, y, z)
        T = np.array([tx, ty, tz])

        mtx = np.eye(4)
        mtx[:3, :3] = self.quaternion_to_rotation_matrix(R)
        mtx[:3, 3] = T
        return mtx

    def update_quaternion_from_eular(self):
        mtx = self.get_quaternion_from_eular()
        if mtx is None:
            return
        self.update_transform_result(mtx)
