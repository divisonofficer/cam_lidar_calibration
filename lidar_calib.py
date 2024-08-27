import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import argparse
import os
import numpy as np
import json
from camera_lidar_calibration import CameraLidarCalibration
import cv2
from lidar_calib_util import LidarCalibUtil
from ui.transform_label import TransformLabel
from dataset_process import DatasetProcess

LIDAR_IMAGE_SCALE = 4


class ImageClickApp:
    def __init__(self, root, args):
        self.root = root
        self.args = args

        self.util = LidarCalibUtil(args)

        self.target = 0
        self.pairs = []
        self.selected_index = None

        """""" """""" """""" """
        폴더 경로 입력 프레임
        """ """""" """""" """"""

        self.folder_path_frame = tk.Frame(root)
        self.folder_path_frame.grid(row=0, column=0, sticky="nsew")
        self.folder_path_label = tk.Label(self.folder_path_frame, text="Folder Path")
        self.folder_path_label.pack(side=tk.LEFT)
        self.folder_path_entry = tk.Entry(self.folder_path_frame)
        self.folder_path_entry.pack(side=tk.LEFT)

        def load_image_from_entry():
            folder_path = filedialog.askdirectory(initialdir=self.args.init_root_path)
            if folder_path:
                self.folder_path_entry.delete(0, tk.END)
                self.folder_path_entry.insert(0, folder_path)
                self.load_image(folder_path)

        self.folder_path_button = tk.Button(
            self.folder_path_frame, text="Load", command=load_image_from_entry
        )
        self.folder_path_button.pack(side=tk.LEFT)

        # 메인 프레임
        main_frame = tk.Frame(root)
        main_frame.grid(row=1, column=0, sticky="nsew")

        # 카메라 캔버스 크기 설정
        self.canvas = tk.Canvas(
            main_frame,
            width=1440,
            height=928 + 128 * 4,
        )
        self.canvas.grid(row=0, column=0, columnspan=2, sticky="nsew")

        # 리스트뷰 및 버튼 프레임
        side_frame = tk.Frame(main_frame)
        side_frame.grid(row=0, column=2, sticky="nsew")

        self.listbox = tk.Listbox(side_frame, width=40)
        self.listbox.pack(side=tk.TOP, fill=tk.Y)
        self.listbox.bind("<<ListboxSelect>>", self.on_listbox_select)

        delete_button = tk.Button(
            side_frame, text="Delete", command=self.delete_selected
        )
        delete_button.pack(pady=5)

        save_button = tk.Button(side_frame, text="Save", command=self.save_pairs)
        save_button.pack(pady=5)

        save_points_button = tk.Button(
            side_frame, text="Save Points", command=self.save_points_ply
        )
        save_points_button.pack(pady=5)

        ### Colorbar Scaling 스크롤 버튼
        self.MAX_LIDAR_RANGE = 10000
        self.MAX_STEREO_DISPARITY = 128

        self.label_lidar_max = tk.Label(
            side_frame, text=f"Max Lidar Range: {self.MAX_LIDAR_RANGE}"
        )
        self.label_lidar_max.pack(pady=5)
        btn_lidar_max_left = tk.Button(
            side_frame, text="Lidar Max -", command=self.lidar_max_minus
        )
        btn_lidar_max_left.pack(pady=5)
        btn_lidar_max_right = tk.Button(
            side_frame, text="Lidar Max +", command=self.lidar_max_plus
        )
        btn_lidar_max_right.pack(pady=5)
        btn_stereo_max_left = tk.Button(
            side_frame, text="Stereo Max -", command=self.stereo_max_minus
        )
        btn_stereo_max_left.pack(pady=5)
        self.label_stereo_max = tk.Label(
            side_frame, text=f"Max Disparity: {self.MAX_STEREO_DISPARITY}"
        )
        self.label_stereo_max.pack(pady=5)

        btn_stereo_max_right = tk.Button(
            side_frame, text="Stereo Max +", command=self.stereo_max_plus
        )
        btn_stereo_max_right.pack(pady=5)

        btn_process_dataset = tk.Button(
            side_frame, text="Process Dataset", command=self.process_dataset
        )
        btn_process_dataset.pack(pady=5)

        self.progressbar_dataset = ttk.Progressbar(
            side_frame, orient=tk.HORIZONTAL, length=500, mode="determinate"
        )
        self.progressbar_dataset.pack(pady=5)

        # 라이다 이미지 스크롤 설정
        self.scroll_x = 0  # 스크롤 위치를 관리하는 변수
        self.view_width = 1024  # 가로의 1/4만 보이게 설정

        # 스크롤바 추가
        scrollbar = tk.Scrollbar(root, orient=tk.HORIZONTAL, command=self.scroll_lidar)
        scrollbar.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.canvas.config(xscrollcommand=scrollbar.set)

        # 키 이벤트 바인딩
        self.canvas.bind_all("<Left>", lambda event: self.move_lidar(-1))
        self.canvas.bind_all("<Right>", lambda event: self.move_lidar(1))

        # 클릭 이벤트 바인딩
        self.canvas.bind("<Button-1>", self.on_click)

        self.transform_label = TransformLabel(root)
        self.transform_label._save_transform = self.save_transform
        self.transform_label._compute_transform = self.compute_transform
        self.transform_label._read_transform = self.read_transform_from_npz
        self.transform_label().grid(row=3, column=0, sticky="nsew")

        self.update_pairs_listview()

        if args.folder_path:
            self.folder_path_entry.insert(0, args.folder_path)
            self.load_image(args.folder_path)

    ##### Colorbar Scaling 스크롤 버튼 함수

    def process_dataset(self):
        mtx = self.transform_label.get_quaternion_from_eular()
        folder = filedialog.askdirectory(initialdir=self.args.init_root_path)

        def update_progressbar(p_idx, msg):
            self.progressbar_dataset.config(value=p_idx)
            print(f"{p_idx}% f{msg}")
            self.progressbar_dataset

        if folder and mtx is not None:
            dataset_process = DatasetProcess(folder)
            folders = dataset_process.search_folder(folder)
            dataset_process.process_scene_list(folders, mtx, update_progressbar)

    def lidar_max_minus(self):
        self.MAX_LIDAR_RANGE -= 1000
        if self.MAX_LIDAR_RANGE < 5000:
            self.MAX_LIDAR_RANGE = 5000
        self.update_lidar_image_by_max()

    def lidar_max_plus(self):
        self.MAX_LIDAR_RANGE += 1000
        self.update_lidar_image_by_max()

    def stereo_max_minus(self):
        self.MAX_STEREO_DISPARITY -= 5
        if self.MAX_STEREO_DISPARITY < 20:
            self.MAX_STEREO_DISPARITY = 20
        self.update_disparity_image_by_max()

    def stereo_max_plus(self):
        self.MAX_STEREO_DISPARITY += 5
        self.update_disparity_image_by_max()

    def update_disparity_image_by_max(self):
        self.label_stereo_max.config(text=f"Max Disparity: {self.MAX_STEREO_DISPARITY}")
        disparity = self.post_data["disparity"]
        dis_ranges = [self.MAX_STEREO_DISPARITY - 20, self.MAX_STEREO_DISPARITY]
        disparity = np.clip(disparity, dis_ranges[0], dis_ranges[1])
        disparity = (disparity - dis_ranges[0]) / (dis_ranges[1] - dis_ranges[0]) * 255
        disparity = disparity.astype(np.uint8)
        disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_MAGMA)
        self.camera_image = Image.fromarray(disparity)
        self.camera_tk_image = ImageTk.PhotoImage(self.camera_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.camera_tk_image)
        self.update_canvas_line()

    def update_lidar_image_by_max(self):
        self.label_lidar_max.config(text=f"Max Lidar Range: {self.MAX_LIDAR_RANGE}")
        ranges_image = self.npz_data["ranges"]
        dis_ranges = [self.MAX_LIDAR_RANGE - 5000, self.MAX_LIDAR_RANGE]
        ranges_image = np.clip(ranges_image, dis_ranges[0], dis_ranges[1])
        ranges_image = (
            (ranges_image - dis_ranges[0]) / (dis_ranges[1] - dis_ranges[0]) * 255
        ).astype(np.uint8)
        ranges_image = cv2.applyColorMap(ranges_image, cv2.COLORMAP_MAGMA)
        self.lidar_image = Image.fromarray(ranges_image)
        self.lidar_image = self.lidar_image.resize(
            (
                self.lidar_image.width * LIDAR_IMAGE_SCALE,
                self.lidar_image.height * LIDAR_IMAGE_SCALE,
            ),
            Image.NEAREST,
        )
        self.lidar_tk_image_full = ImageTk.PhotoImage(self.lidar_image)
        self.update_lidar_image()

    def read_pairs(self, pairs):
        self.pairs = []
        if hasattr(self, "listbox"):
            self.listbox.delete(0, tk.END)
        for pair in pairs:
            self.pairs.append(
                {
                    "camera": (pair[0], pair[1]),
                    "lidar": (pair[2], pair[3]),
                }
            )

    def update_pairs_listview(self):
        for pair in self.pairs:
            self.listbox.insert(
                tk.END,
                f"Pair: Camera ({pair['camera']}), Lidar ({pair['lidar']})",
            )

    def load_camera_image(self):
        folder_path = self.args.folder_path
        # raw.npz 파일이 있는지 확인
        raw_npz_path = os.path.join(folder_path, "raw.npz")
        if os.path.exists(raw_npz_path):
            data = np.load(raw_npz_path)
            self.npz_data = data

        if os.path.exists(raw_npz_path.replace("raw", "post")):
            data = np.load(raw_npz_path.replace("raw", "post"))
            self.post_data = data

        if os.path.exists(raw_npz_path.replace("raw", "pairs")):
            data = np.load(raw_npz_path.replace("raw", "pairs"))
            pairs = data["pairs"]
            self.read_pairs(pairs)
        else:
            self.pairs = []

        # disparity.png가 있는지 확인
        disparity_path = os.path.join(folder_path, "disparity.png")
        if os.path.exists(disparity_path):
            return Image.open(disparity_path)

        # left_tonemapped.png가 있는지 확인
        left_tonemapped_path = os.path.join(folder_path, "left_tonemapped.png")
        if os.path.exists(left_tonemapped_path):
            return Image.open(left_tonemapped_path)

        if os.path.exists(raw_npz_path):
            left_image = data["left.npy"]
            # 이미지를 (width x height x 3) 형식으로 만들어서 중간 채널 가져오기
            return Image.fromarray(left_image[:, :, 1])

        raise FileNotFoundError(
            "No valid camera image file found in the specified folder."
        )

    def load_lidar_image(self):
        folder_path = self.args.folder_path

        # raw.npz 파일에서 ranges.npy를 읽어들임
        raw_npz_path = os.path.join(folder_path, "raw.npz")
        if os.path.exists(raw_npz_path):
            data = np.load(raw_npz_path)
            ranges_image = data["ranges.npy"]
            # uint16를 uint8로 스케일 다운
            ranges_image_scaled = (ranges_image / 128).astype(np.uint8)
            ranges_image_scaled = cv2.applyColorMap(
                ranges_image_scaled, cv2.COLORMAP_MAGMA
            )
            lidar_image = Image.fromarray(ranges_image_scaled)
            # 해상도를 4배로 키우기
            lidar_image = lidar_image.resize(
                (
                    lidar_image.width * LIDAR_IMAGE_SCALE,
                    lidar_image.height * LIDAR_IMAGE_SCALE,
                ),
                Image.NEAREST,
            )
            return lidar_image

        raise FileNotFoundError(
            "No valid lidar image file found in the specified folder."
        )

    def update_lidar_image(self):
        # 라이다 이미지를 스크롤 위치에 따라 크롭하여 표시, 높이는 원본 그대로 유지
        cropped_lidar_image = self.lidar_image.crop(
            (self.scroll_x, 0, self.scroll_x + self.view_width, self.lidar_image.height)
        )
        self.lidar_tk_image = ImageTk.PhotoImage(cropped_lidar_image)

        # 라이다 이미지를 캔버스에 그리기
        self.canvas.create_image(
            0, self.camera_tk_image.height(), anchor=tk.NW, image=self.lidar_tk_image
        )

        self.update_canvas_line()

    def update_canvas_line(self):
        # 모든 선 다시 그리기
        self.canvas.delete("line")
        for i, pair in enumerate(self.pairs):
            self.draw_line(
                pair["camera"],
                pair["lidar"],
                highlight=(i == self.selected_index),
            )

    def scroll_lidar(self, *args):
        if args[0] == "scroll":
            direction = int(args[1])
            self.move_lidar(direction)

    def move_lidar(self, direction):
        # 스크롤 위치 변경, 방향키로 이동
        new_scroll_x = self.scroll_x + direction * self.view_width // 2

        # 경계 체크
        if 0 <= new_scroll_x <= self.lidar_image.width - self.view_width:
            self.scroll_x = new_scroll_x
            self.update_lidar_image()

    def on_click(self, event):
        if self.target == 0:
            # 카메라 이미지 클릭
            if (
                0 <= event.x < self.camera_tk_image.width()
                and 0 <= event.y < self.camera_tk_image.height()
            ):
                camera_x, camera_y = event.x, event.y
                self.pairs.append({"camera": (camera_x, camera_y)})
                self.listbox.insert(tk.END, f"Camera: ({camera_x}, {camera_y})")
                self.target = 1
        elif self.target == 1:
            # 라이다 이미지 클릭
            if (
                0 <= event.x < self.view_width
                and self.camera_tk_image.height()
                <= event.y
                < self.camera_tk_image.height() + self.lidar_tk_image.height()
            ):
                lidar_x, lidar_y = (
                    event.x + self.scroll_x,
                    event.y - self.camera_tk_image.height(),
                )
                lidar_x //= LIDAR_IMAGE_SCALE
                lidar_y //= LIDAR_IMAGE_SCALE
                self.pairs[-1]["lidar"] = (lidar_x, lidar_y)
                self.listbox.delete(tk.END)
                self.listbox.insert(
                    tk.END,
                    f"Pair: Camera ({self.pairs[-1]['camera']}), Lidar ({lidar_x}, {lidar_y})",
                )
                self.draw_line(
                    self.pairs[-1]["camera"], self.pairs[-1]["lidar"], highlight=True
                )
                self.target = 0

    def draw_line(self, camera_coords, lidar_coords, highlight=False):
        cam_x, cam_y = camera_coords
        lidar_coords = (
            lidar_coords[0] * LIDAR_IMAGE_SCALE,
            lidar_coords[1] * LIDAR_IMAGE_SCALE,
        )
        lid_x, lid_y = (
            lidar_coords[0] - self.scroll_x,
            lidar_coords[1] + self.camera_tk_image.height(),
        )
        color = "green" if highlight else "red"
        # if 0 <= lid_x < self.view_width:  # 라이다 이미지 내에 있는 경우만 그리기
        self.canvas.create_line(
            cam_x, cam_y, lid_x, lid_y, fill=color, width=2, tags="line"
        )

    def on_listbox_select(self, event):
        if not self.listbox.curselection():
            return

        index = self.listbox.curselection()[0]
        if index < len(self.pairs):
            self.selected_index = index
            self.canvas.delete("all")
            # 이미지 다시 그리기
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.camera_tk_image)
            self.update_lidar_image()

    def delete_selected(self):
        if self.selected_index is not None:
            del self.pairs[self.selected_index]
            self.listbox.delete(self.selected_index)
            self.selected_index = None
            self.canvas.delete("all")
            # 이미지 다시 그리기
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.camera_tk_image)
            self.update_lidar_image()

    def save_pairs(self):
        if not self.pairs:
            messagebox.showinfo("No Data", "There are no pairs to save.")
            return

        pair_np = np.array(
            [
                [
                    pair["camera"][0],
                    pair["camera"][1],
                    pair["lidar"][0],
                    pair["lidar"][1],
                ]
                for pair in self.pairs
            ]
        )
        save_path = self.args.folder_path + "/pairs.npz"
        np.savez(save_path, pairs=pair_np)
        messagebox.showinfo("Saved", f"Pairs have been saved to {save_path}")

    def save_transform(self, transform: np.ndarray):
        post_npz = np.load(self.args.folder_path + "/post.npz")
        post_npz = {**post_npz, "transform": transform}
        np.savez(self.args.folder_path + "/post.npz", **post_npz)
        messagebox.showinfo("Saved", "Transform has been saved.")

    def read_transform_from_npz(self):
        """
        Read lidar transform mtx from post.npz if it exists
        """
        post_npz = np.load(self.args.folder_path + "/post.npz")
        if "transform" in post_npz:
            init_mtx = post_npz["transform"]
            return init_mtx
        return None

    def read_camera_intrinsic(self):
        """
        Read left camera intrinsic, stereo baseline, and disparity map from post.npz
        """
        post_npz = np.load(self.args.folder_path + "/post.npz")
        k_left = post_npz["k_left"]
        baseline = np.linalg.norm(post_npz["T"])
        disparity = post_npz["disparity"]
        depth = post_npz["depth"]
        d_left = post_npz["d_left"]
        k_right = post_npz["k_right"]
        d_right = post_npz["d_right"]
        R = post_npz["R"]
        T = post_npz["T"]

        return k_left, baseline, disparity, depth, d_left, k_right, d_right, R, T

    def read_stereo_depth(self):
        post_npz = np.load(self.args.folder_path + "/post.npz")
        return post_npz["depth"]

    def compute_transform(self, use_pair=True, use_open3d=False):
        pairs_npz_path = self.args.folder_path + "/pairs.npz"
        pairs = (
            np.load(pairs_npz_path)["pairs"]
            if use_pair and os.path.exists(pairs_npz_path)
            else None
        )
        init_mtx = None
        if not use_pair or pairs is None:
            init_mtx = self.transform_label.get_quaternion_from_eular()
            if init_mtx is None:
                init_mtx = self.read_transform_from_npz()
        k_left, baseline, disparity, depth, d_left, k_right, d_right, R, T = (
            self.read_camera_intrinsic()
        )
        points = self.npz_data["points"]

        calibration = CameraLidarCalibration(
            self.args,
            disparity,
            points,
            k_left,
            baseline,
            pairs,
            init_mtx,
            max_depth=self.transform_label.get_depth_max_option(),
            remap_mask=self.remap_mask if hasattr(self, "remap_mask") else None,
        )
        if not hasattr(self, "remap_mask"):
            self.remap_mask = calibration.compute_remap_mask(
                disparity.shape[0],
                disparity.shape[1],
                k_left,
                d_left,
                k_right,
                d_right,
                R,
                T,
            )
        transform = calibration.estimate_transform(use_open3d)

        self.transform_label.update_transform_result(transform)

        transform_img, plot_list = calibration.render_transform_lidarpoints_to_image(
            points, depth
        )

        cv2.imwrite(self.args.folder_path + "/transform_image.png", transform_img)
        transform_image = Image.fromarray(transform_img[:, :, ::-1])
        transform_image.resize(
            (int(transform_image.width * 0.75), int(transform_image.height * 0.75))
        )
        tkImage = ImageTk.PhotoImage(transform_image)
        self.transform_label.transform_result_image_view.configure(image=tkImage)
        self.transform_label.transform_result_image_view.image = tkImage

    def save_points_ply(self):
        self.util.store_point_cloud(self.npz_data["points"] * 1000, "points")
        k_left = self.post_data["k_left"]
        self.util.store_point_cloud(
            self.util.depth2points(
                self.post_data["depth"], k_left[0, 0], k_left[0, 2], k_left[1, 2]
            ),
            "camera_points",
        )

    def load_image(self, folder_path: str):
        self.args.folder_path = folder_path
        self.camera_image = self.load_camera_image()
        self.lidar_image = self.load_lidar_image()

        self.camera_tk_image = ImageTk.PhotoImage(self.camera_image)
        self.lidar_tk_image_full = ImageTk.PhotoImage(self.lidar_image)
        # 카메라 이미지 배치
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.camera_tk_image)
        # 캔버스 위에 라이다 이미지 부분을 그리기 위한 초기 이미지 크롭
        self.update_lidar_image()

        if hasattr(self, "listbox"):
            self.update_pairs_listview()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display camera and lidar images and collect click coordinates."
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        required=False,
        help="Path to the folder containing the image files",
    )
    parser.add_argument(
        "--init_root_path",
        type=str,
        default="/bean/lucid",
        help="Initial root path for file dialog",
    )

    args = parser.parse_args()

    root = tk.Tk()
    root.title("Camera and Lidar Image Click Coordinates")

    app = ImageClickApp(root, args)

    root.mainloop()
