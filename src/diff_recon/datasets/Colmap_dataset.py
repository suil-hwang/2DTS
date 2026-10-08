import os
import math
from concurrent.futures import ThreadPoolExecutor
from copy import copy
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from argparse import Namespace

from .colmap_loader import read_points3D_binary, CameraInfo
from .Base_dataset import BaseDatasetFactory
from .dataset_utils import getCameraExtent, read_validated_colmap_cameras
from ..models.point_cloud import PointCloud
from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.file_handler import LocalHandler, BaseFileHandler
from ..utils.camera import Camera, getWorld2ViewMatrix


_DEFAULT_BACKGROUND = object()


def solve_target_res(target_res: int | list[int] | None, orig_w: int, orig_h: int) -> tuple[int, int]:
    w, h = orig_w, orig_h

    if target_res is None:
        if w >= h and w > 1600:
            w, h = 1600, 1600 * orig_h // orig_w
        elif w < h and h > 1600:
            w, h = 1600 * orig_w // orig_h, 1600
    elif isinstance(target_res, int):
        if target_res <= 0:
            target_res = 1
        w, h = orig_w // target_res, orig_h // target_res
    elif isinstance(target_res, list):
        if len(target_res) != 2 or any(not isinstance(value, (int, np.integer)) or value <= 0 for value in target_res):
            raise ValueError("target_res dimensions must be two positive integers")
        w, h = target_res
    else:
        raise ValueError("target_res must be either an int of scale ratio or a list of [width, height]")

    return max(1, w), max(1, h)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Unsigned 8/16-bit (H, W, C) image -> float32 (C, H, W) in [0, 1]."""
    if image.dtype not in (np.uint8, np.uint16) or image.ndim != 3:
        raise ValueError("Images must have unsigned 8/16-bit (height, width, channels) values")
    return np.ascontiguousarray(image.astype(np.float32).transpose(2, 0, 1) / np.iinfo(image.dtype).max)


class ColmapDataset(Dataset):
    def __init__(
        self,
        file_handler: BaseFileHandler,
        cam_infos: list[CameraInfo],
        target_res: int | list[int] = None,
        background: str = None,
        znear: float = 1.0,
        prefetch: bool = False,
        neighbor_cams_args: Namespace = None,
    ):
        super().__init__()
        self.file_handler = file_handler
        self.cam_infos = cam_infos
        self.target_res = target_res
        self.znear = znear
        self.background = background
        self._image_sizes = {}

        if prefetch:
            self._prefetched_imgs = [self._get_image(cam_info.image_path) for cam_info in self.cam_infos]
        if neighbor_cams_args is not None:
            self._neighbor_cams = self._get_neighbor_graph(neighbor_cams_args)

    def _get_neighbor_graph(self, args) -> list[list[int]]:
        neighbor_cams = []
        centers = np.array([-cam_info.R @ cam_info.T for cam_info in self.cam_infos])
        dirs = np.array([cam_info.R[:, 2] for cam_info in self.cam_infos])
        for i in range(len(self.cam_infos)):
            dists = np.linalg.norm(centers[i] - centers, axis=1)
            angles = np.arccos((dirs[i] @ dirs.T).clip(-1, 1)) / np.pi * 180
            sorted_ids = np.lexsort((angles, dists))

            mask = (angles[sorted_ids] < args.max_angle) & (dists[sorted_ids] > args.min_dist) & (dists[sorted_ids] < args.max_dist)
            mask &= sorted_ids != i  # exclude itself
            sorted_ids = sorted_ids[mask][: args.max_n]
            neighbor_cams.append(sorted_ids.tolist())
        return neighbor_cams

    def _get_bg_color(self) -> np.ndarray:
        if self.background is None:
            bg_color = None
        elif self.background == "white":
            bg_color = np.array([1, 1, 1])
        elif self.background == "black":
            bg_color = np.array([0, 0, 0])
        elif self.background == "random":
            bg_color = np.random.rand(3)
        else:
            raise ValueError("dataset background must be either 'white', 'black', 'random' or None")
        return bg_color

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load RGB/RGBA or unsigned 16-bit grayscale as an (H, W, 3/4) color array."""
        if image_path is None:
            return None
        image_key = image_path.replace("\\", "/")

        # some dataset's images are stored with a different structure
        if not self.file_handler.hasFile(image_path):
            path_list = image_key.split("/")
            img_dir = "_".join(path_list[-1].split("_")[:2])
            path_list = path_list[:-1] + [img_dir, path_list[-1]]
            image_path = "/".join(path_list)

        with Image.open(self.file_handler.getFilePath(image_path)) as image:
            self._image_sizes[image_key] = (image.width, image.height)
            img_size = solve_target_res(self.target_res, image.width, image.height)
            uint16_gray = image.mode.startswith("I;16") or (image.mode == "I" and image.format == "PNG")
            if uint16_gray:
                image_array = np.array(image.resize(img_size, Image.Resampling.BILINEAR), dtype=np.uint16)
                image_array = np.repeat(image_array[..., None], 3, axis=-1)
            else:
                if image.mode in ("I", "F"):
                    raise ValueError(f"Unsupported image mode {image.mode}; use RGB/RGBA or unsigned 16-bit grayscale")
                mode = "RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB"
                image_array = np.array(image.convert(mode).resize(img_size, Image.Resampling.BILINEAR))
        return image_array

    def _get_image(self, image_path: str) -> np.ndarray:
        """Load, resize and normalize the image: float32 array of shape (3, H, W) or (4, H, W) in [0, 1]."""
        image = self._load_image(image_path)
        return None if image is None else normalize_image(image)

    def __len__(self):
        return len(self.cam_infos)

    def _get_item(self, idx: int, gt_image_array: np.ndarray = None, bg_color=_DEFAULT_BACKGROUND) -> Camera:
        cam_info: CameraInfo = self.cam_infos[idx]
        if gt_image_array is None:
            gt_image_array = self._prefetched_imgs[idx] if hasattr(self, "_prefetched_imgs") else self._get_image(cam_info.image_path)
        if bg_color is _DEFAULT_BACKGROUND:
            bg_color = self._get_bg_color()

        if gt_image_array is not None and gt_image_array.shape[0] == 4:
            gt_alpha_mask = gt_image_array[3]
            gt_image_array = gt_image_array[:3]
            if bg_color is not None:
                gt_image_array = gt_image_array * gt_alpha_mask
                gt_image_array += bg_color.reshape(3, 1, 1) * (1 - gt_alpha_mask)
        else:
            gt_alpha_mask = None

        fovy = cam_info.FovY
        if fovy is None and cam_info.image_path is not None:
            original_size = self._image_sizes.get(cam_info.image_path.replace("\\", "/"))
            if original_size is not None:
                fovy = 2 * math.atan(math.tan(cam_info.FovX / 2) * original_size[1] / original_size[0])
        camera = Camera(
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FovX,
            FoVy=fovy,
            image_width=cam_info.width if gt_image_array is None else None,
            image_height=cam_info.height if gt_image_array is None else None,
            gt_image=gt_image_array,
            gt_alpha_mask=gt_alpha_mask,
            image_name=cam_info.image_name,
            camera_id=cam_info.camera_id,
            uid=idx,
            znear=self.znear,
            bg_color=bg_color,
        )
        return camera

    def __getitem__(self, idx) -> Camera:
        bg_color = self._get_bg_color()
        camera = self._get_item(idx, bg_color=bg_color)
        if hasattr(self, "_neighbor_cams") and len(self._neighbor_cams[idx]) > 0:
            neighbor_cam_id = np.random.choice(self._neighbor_cams[idx])
            camera.neighbor_cam = self._get_item(neighbor_cam_id, bg_color=bg_color)
        return camera


class DeviceColmapDataset(Dataset):
    """A ColmapDataset served from `device`: each image is decoded once and kept as unsigned integers."""

    def __init__(self, dataset: ColmapDataset, device: torch.device):
        if hasattr(dataset, "_neighbor_cams"):
            raise ValueError("cache_on_device does not support neighbor cameras")
        self.background = dataset.background
        self.device = device
        # uint8 -> [0, 1] exactly as normalize_image computes it; CUDA divides by a scalar through its reciprocal
        self.unit_values = torch.from_numpy(normalize_image(np.arange(256, dtype=np.uint8).reshape(1, -1, 1)).ravel()).to(device)

        def load(idx: int):  # decode and build the camera on the CPU; images are independent, so threads overlap them
            image = dataset._load_image(dataset.cam_infos[idx].image_path)
            camera = dataset._get_item(idx, None if image is None else normalize_image(image))
            camera.gt_image = camera.alpha_mask = None  # redrawn per item from the uint8 image
            return image, camera

        with ThreadPoolExecutor() as pool:
            loaded = list(pool.map(load, range(len(dataset))))
        self.unit_values_16bit = None
        if any(image is not None and image.dtype == np.uint16 for image, _ in loaded):
            self.unit_values_16bit = torch.from_numpy(normalize_image(np.arange(65536, dtype=np.uint16).reshape(1, -1, 1)).ravel()).to(device)
        self.cameras = [camera.to(device) for _, camera in loaded]
        self.images = [None if image is None else torch.from_numpy(image).to(device).permute(2, 0, 1) for image, _ in loaded]

    def __len__(self):
        return len(self.cameras)

    def _get_bg_color(self) -> torch.Tensor:
        if self.background is None:
            return None
        if self.background == "random":
            return torch.rand(3, dtype=torch.float64, device=self.device)
        return torch.full((3,), {"white": 1.0, "black": 0.0}[self.background], dtype=torch.float64, device=self.device)

    def __getitem__(self, idx: int) -> Camera:
        camera = copy(self.cameras[idx])
        bg_color = self._get_bg_color()
        camera.bg_color = None if bg_color is None else bg_color.float()
        if self.images[idx] is None:
            return camera
        unit_values = self.unit_values_16bit if self.images[idx].dtype == torch.uint16 else self.unit_values
        gt_image = unit_values[self.images[idx].int()]
        alpha_mask = None
        if gt_image.shape[0] == 4:
            gt_image, alpha_mask = gt_image[:3], gt_image[3]
            if bg_color is not None:
                gt_image = ((gt_image * alpha_mask).double() + bg_color.view(3, 1, 1) * (1 - alpha_mask).double()).float()
        camera.gt_image = gt_image.clamp(0.0, 1.0)
        camera.alpha_mask = alpha_mask
        return camera


class ColmapDatasetFactory(BaseDatasetFactory):
    def __init__(self, config: Config = None, logger: Logger = None):
        super().__init__(config, logger)
        config = self._config
        train_target_res = config.train_target_res
        test_target_res = config.test_target_res
        hold_test_set = config.hold_test_set
        background = config.background
        test_background = config.test_background if config.test_background is not None else background
        prefetch = config.prefetch if config.prefetch is not None else False
        neighbor_cams_args = config.neighbor_cams_args if config.neighbor_cams_args is not None else None

        self._file_handler = self._get_file_handler()

        train_cam_infos, test_cam_infos = self.getCameraInfos()
        if not hold_test_set:
            train_cam_infos += test_cam_infos
            self._logger.warning(f"hold_test_set not set, will merge test set into train set")
        if not train_cam_infos:
            raise ValueError("Training set is empty; check the registered cameras and holdout split")
        self._logger.info(f"Train set size: {len(train_cam_infos)}, Test set size: {len(test_cam_infos)}")

        self.cameras_extent = getCameraExtent(train_cam_infos)
        self.znear = self.cameras_extent / 1000
        self._logger.info(f"Camera extent: {self.cameras_extent:.2f}, znear set to: {self.znear:.5f}")
        if prefetch:
            self._logger.info("Prefetching data into memory...")

        if neighbor_cams_args is not None:
            self._logger.info("Loading neighbor cameras")
        self._train_dataset = ColmapDataset(self._file_handler, train_cam_infos, train_target_res, background, self.znear, prefetch, neighbor_cams_args)
        self._test_dataset = ColmapDataset(self._file_handler, test_cam_infos, test_target_res, test_background, self.znear, prefetch)

        if len(self._train_dataset) > 0:
            train_cam = self._train_dataset[0]
            self._logger.info(f"Dataset contains alpha mask: {train_cam.alpha_mask is not None}.")
            self._logger.info(f"Train resolution: {train_cam.image_width}x{train_cam.image_height}")
        if len(self._test_dataset) > 0:
            test_cam = self._test_dataset[0]
            self._logger.info(f"Test resolution: {test_cam.image_width}x{test_cam.image_height}")

    def _get_file_handler(self) -> BaseFileHandler:
        if self._config.local_dir is None:
            raise ValueError("local_dir must be set in the config")
        dataset_path = os.path.join(self._config.local_dir, self._config.scene_id) if self._config.scene_id else self._config.local_dir
        return LocalHandler(dataset_path)

    def _getCameraInfos(self) -> tuple[list[CameraInfo], list[CameraInfo]]:
        fs = self._file_handler
        images_bin_path = "sparse/0/images.bin"
        images_txt_path = "sparse/0/images.txt"
        cameras_bin_path = "sparse/0/cameras.bin"
        cameras_txt_path = "sparse/0/cameras.txt"
        images_folder = "images"

        if fs.hasFile(images_bin_path):
            images_path = fs.getFilePath(images_bin_path)
            self._logger.info(f"Fetching extrinsics data from {images_bin_path}.")
        elif fs.hasFile(images_txt_path):
            images_path = fs.getFilePath(images_txt_path)
            self._logger.info(f"Fetching extrinsics data from {images_txt_path}.")
        else:
            raise FileNotFoundError(f"Cannot find {images_bin_path} or {images_txt_path}")

        if fs.hasFile(cameras_bin_path):
            cameras_path = fs.getFilePath(cameras_bin_path)
            self._logger.info(f"Fetching intrinsics data from {cameras_bin_path}.")
        elif fs.hasFile(cameras_txt_path):
            cameras_path = fs.getFilePath(cameras_txt_path)
            self._logger.info(f"Fetching intrinsics data from {cameras_txt_path}.")
        else:
            raise FileNotFoundError(f"Cannot find {cameras_bin_path} or {cameras_txt_path}")

        cam_infos = read_validated_colmap_cameras(images_path, cameras_path, images_folder)
        cam_infos = sorted(cam_infos, key=lambda x: x.image_name)

        hold_interval = self._config.hold_interval if self._config.hold_interval is not None else 8
        if not isinstance(hold_interval, (int, np.integer)) or isinstance(hold_interval, bool) or hold_interval <= 0:
            raise ValueError("hold_interval must be a positive integer")
        train_cam_infos = [cam for i, cam in enumerate(cam_infos) if i % hold_interval != 0]
        test_cam_infos = [cam for i, cam in enumerate(cam_infos) if i % hold_interval == 0]
        return train_cam_infos, test_cam_infos

    def _getPointCloud(self) -> PointCloud:
        fs = self._file_handler
        pcd_path = self._config.pcd_path

        if pcd_path is None:
            return PointCloud()

        pcd_path = fs.getFilePath(pcd_path)
        self._logger.info(f"Fetching point cloud data from {pcd_path}.")

        if pcd_path.endswith(".bin"):
            xyz, rgb, _ = read_points3D_binary(pcd_path)
            pcd = PointCloud(xyz, rgb / 255.0)
        elif pcd_path.endswith(".ply"):
            pcd = PointCloud().fetchPly(pcd_path)
        else:
            raise ValueError(f"Unsupported point cloud file format: {pcd_path.split('.')[-1]}")

        return pcd

    def cacheOnDevice(self, device: torch.device):
        """Serve train and test cameras from `device` (DeviceColmapDataset) through worker-free loaders."""
        train_dataset = DeviceColmapDataset(self._train_dataset, device)
        test_dataset = DeviceColmapDataset(self._test_dataset, device)
        self._train_dataset, self._test_dataset = train_dataset, test_dataset
        self._num_workers = 0
        self._pin_memory = False
        for name in ("_train_loader", "_test_loader", "_train_dataloader"):
            self.__dict__.pop(name, None)

    def getCameraInfos(self) -> tuple[list[CameraInfo], list[CameraInfo]]:
        if not hasattr(self, "_cam_infos") or self._cam_infos is None:
            self._cam_infos = self._getCameraInfos()
        return self._cam_infos

    def getPointCloud(self) -> PointCloud:
        if not hasattr(self, "_point_cloud") or self._point_cloud is None:
            self._point_cloud = self._getPointCloud()
        return self._point_cloud

    def getSceneInfo(self) -> dict:
        return None
