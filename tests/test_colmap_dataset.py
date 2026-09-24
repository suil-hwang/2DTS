import math
import struct
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from src.diff_recon.datasets.Colmap_dataset import ColmapDataset, ColmapDatasetFactory, DeviceColmapDataset
from src.diff_recon.datasets.colmap_loader import CameraInfo, read_extrinsics_binary, read_extrinsics_text, readColmapCameras
from src.diff_recon.utils.file_handler import LocalHandler


def make_cam(center, R=np.eye(3)) -> CameraInfo:
    R = np.asarray(R, dtype=np.float64)
    T = -R.T @ np.asarray(center, dtype=np.float64)
    return CameraInfo(camera_id=1, R=R, T=T, FovY=0.8, FovX=0.8, image_path=None, image_name="cam", width=8, height=6)


class NeighborGraphTest(unittest.TestCase):
    def neighbors(self, cams, **kwargs) -> list[list[int]]:
        args = Namespace(**{"max_angle": 30, "min_dist": 0.01, "max_dist": 10.0, "max_n": 8, **kwargs})
        return ColmapDataset(None, cams, neighbor_cams_args=args)._neighbor_cams

    def test_excludes_only_the_camera_itself(self):
        # Regression: self-exclusion indexed the distance-sorted order, so camera i lost its i-th closest neighbor
        cams = [make_cam([x, 0.0, 0.0]) for x in range(4)]
        self.assertEqual(self.neighbors(cams), [[1, 2, 3], [0, 2, 3], [1, 3, 0], [2, 1, 0]])

    def test_applies_distance_angle_and_count_limits(self):
        cams = [
            make_cam([0.0, 0.0, 0.0]),
            make_cam([2.0, 0.0, 0.0]),
            make_cam([1.0, 0.0, 0.0], Rotation.from_euler("y", 20, degrees=True).as_matrix()),
            make_cam([0.005, 0.0, 0.0]),  # closer than min_dist
            make_cam([20.0, 0.0, 0.0]),  # farther than max_dist
            make_cam([0.5, 0.0, 0.0], Rotation.from_euler("y", 45, degrees=True).as_matrix()),  # beyond max_angle
        ]
        self.assertEqual(self.neighbors(cams)[0], [2, 1])
        self.assertEqual(self.neighbors(cams, max_n=1)[0], [2])


def write_images_text(path: Path, images: list[dict]) -> None:
    lines = []
    for image in images:
        lines.append(" ".join(map(repr, [image["id"], *image["qvec"], *image["tvec"], image["camera_id"]])) + f" {image['name']}")
        lines.append(" ".join(f"{x!r} {y!r} {pid}" for (x, y), pid in zip(image["xys"], image["point3D_ids"])))
    path.write_text("\n".join(lines) + "\n")


def write_images_binary(path: Path, images: list[dict]) -> None:
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for image in images:
            fid.write(struct.pack("<idddddddi", image["id"], *image["qvec"], *image["tvec"], image["camera_id"]))
            fid.write(image["name"].encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", len(image["point3D_ids"])))
            for (x, y), pid in zip(image["xys"], image["point3D_ids"]):
                fid.write(struct.pack("<ddq", x, y, pid))


class ColmapBinaryModelTest(unittest.TestCase):
    def setUp(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.root = Path(tmp_dir.name)

        rng = np.random.default_rng(0)
        qvecs = Rotation.random(3, random_state=0).as_quat(canonical=True, scalar_first=True)
        self.images = [
            {
                "id": image_id,
                "qvec": qvec.tolist(),
                "tvec": rng.normal(size=3).tolist(),
                "camera_id": 1,
                "name": name,
                "xys": (rng.random((num_points, 2)) * 800).tolist(),
                "point3D_ids": rng.integers(-1, 2**40, size=num_points).tolist(),
            }
            for image_id, qvec, name, num_points in zip([3, 1, 7], qvecs, ["b.png", "a.png", "c.png"], [5, 0, 1])
        ]
        write_images_text(self.root / "images.txt", self.images)
        write_images_binary(self.root / "images.bin", self.images)
        (self.root / "cameras.txt").write_text("1 PINHOLE 800 600 400 410 400 300\n")
        with open(self.root / "cameras.bin", "wb") as fid:
            fid.write(struct.pack("<QiiQQdddd", 1, 1, 1, 800, 600, 400.0, 410.0, 400.0, 300.0))

    def test_binary_extrinsics_match_the_written_images(self):
        images = read_extrinsics_binary(str(self.root / "images.bin"))

        self.assertEqual(list(images), [3, 1, 7])
        for image in self.images:
            parsed = images[image["id"]]
            self.assertEqual((parsed.name, parsed.camera_id), (image["name"], image["camera_id"]))
            np.testing.assert_array_equal(parsed.qvec, image["qvec"])
            np.testing.assert_array_equal(parsed.tvec, image["tvec"])
            self.assertEqual(parsed.xys.shape, (len(image["xys"]), 2))
            np.testing.assert_array_equal(parsed.xys.reshape(-1, 2), np.reshape(image["xys"], (-1, 2)))
            self.assertEqual(parsed.point3D_ids.tolist(), image["point3D_ids"])

    def test_truncated_binary_extrinsics_raise(self):
        path = self.root / "images.bin"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(EOFError):
            read_extrinsics_binary(str(path))

    def test_binary_and_text_extrinsics_agree(self):
        binary = read_extrinsics_binary(str(self.root / "images.bin"))
        text = read_extrinsics_text(str(self.root / "images.txt"))

        self.assertEqual(list(binary), list(text))
        for image_id in text:
            for field in ("qvec", "tvec", "xys", "point3D_ids"):
                np.testing.assert_array_equal(getattr(binary[image_id], field), getattr(text[image_id], field), err_msg=field)

    def test_readColmapCameras_reads_binary_and_text_models_identically(self):
        binary = readColmapCameras(str(self.root / "images.bin"), str(self.root / "cameras.bin"), "images")
        text = readColmapCameras(str(self.root / "images.txt"), str(self.root / "cameras.txt"), "images")

        self.assertEqual([cam.image_name for cam in binary], ["b", "a", "c"])
        for cam_bin, cam_txt in zip(binary, text):
            np.testing.assert_array_equal(cam_bin.R, cam_txt.R)
            np.testing.assert_array_equal(cam_bin.T, cam_txt.T)
            self.assertEqual((cam_bin.FovX, cam_bin.FovY), (cam_txt.FovX, cam_txt.FovY))
        self.assertAlmostEqual(binary[0].FovY, 2 * math.atan(300 / 410))


class DeviceColmapDatasetTest(unittest.TestCase):
    """DeviceColmapDataset must hand out exactly what ColmapDataset does for the same background."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        rng = np.random.default_rng(0)
        Image.fromarray(rng.integers(0, 256, (6, 8, 4), dtype=np.uint8), "RGBA").save(root / "rgba.png")
        Image.fromarray(rng.integers(0, 256, (6, 8, 3), dtype=np.uint8), "RGB").save(root / "rgb.png")
        self.handler = LocalHandler(str(root))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tearDown(self):
        self.tmp.cleanup()

    def datasets(self, background, image_paths=("rgba.png",)):
        cams = [make_cam([0.5 * i, 0.0, -2.0])._replace(image_path=path, width=None, height=None) for i, path in enumerate(image_paths)]
        dataset = ColmapDataset(self.handler, cams, background=background, znear=0.01)
        return dataset, DeviceColmapDataset(dataset, self.device)

    def assert_same_item(self, dataset, device_dataset, bg_color=None):
        if bg_color is not None:  # both draw this background
            dataset._get_bg_color = lambda: bg_color
            device_dataset._get_bg_color = lambda: torch.tensor(bg_color, device=self.device)
        expected, actual = dataset[0], device_dataset[0]
        for name in ("gt_image", "alpha_mask", "bg_color", "world_view_transform", "projection_matrix", "full_proj_transform", "camera_center"):
            value, reference = getattr(actual, name), getattr(expected, name)
            if reference is None:
                self.assertIsNone(value, name)
                continue
            self.assertEqual(value.device.type, self.device.type, name)
            self.assertTrue(torch.equal(value.cpu(), reference), name)
        self.assertEqual((actual.image_width, actual.image_height, actual.tan_fovx, actual.tan_fovy),
                         (expected.image_width, expected.image_height, expected.tan_fovx, expected.tan_fovy))

    def test_items_match_colmap_dataset_bit_for_bit(self):
        for image_path in ("rgba.png", "rgb.png"):
            for background in (None, "white", "black"):
                with self.subTest(image=image_path, background=background):
                    self.assert_same_item(*self.datasets(background, (image_path,)))
            with self.subTest(image=image_path, background="random"):
                self.assert_same_item(*self.datasets("random", (image_path,)), bg_color=np.random.default_rng(1).random(3))

    def test_random_background_is_redrawn_per_item(self):
        _, device_dataset = self.datasets("random")
        first, second = device_dataset[0].bg_color, device_dataset[0].bg_color
        self.assertFalse(torch.equal(first, second))
        self.assertTrue(((first >= 0) & (first < 1)).all().item())

    def test_factory_serves_device_cameras_without_workers(self):
        factory = ColmapDatasetFactory.__new__(ColmapDatasetFactory)
        factory._train_dataset, _ = self.datasets("random", ("rgba.png", "rgb.png"))
        factory._test_dataset, _ = self.datasets("white")
        factory._num_workers, factory._pin_memory = 10, True
        factory.cacheOnDevice(self.device)

        uids = [factory.nextTrainData().uid for _ in range(6)]  # three epochs of two views
        self.assertEqual([sorted(uids[i : i + 2]) for i in range(0, 6, 2)], [[0, 1]] * 3)
        test_cams = list(factory.getTestDataset())
        self.assertEqual(len(test_cams), 1)
        self.assertEqual(test_cams[0].gt_image.device.type, self.device.type)


if __name__ == "__main__":
    unittest.main()
