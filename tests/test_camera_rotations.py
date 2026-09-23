import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.diff_recon.datasets.colmap_loader import CameraInfo, readColmapCameras
from src.diff_recon.datasets.dataset_utils import camInfosToColmap, interpolateCameraInfos
from src.diff_recon.utils.camera import qvec2rotmat, rotmat2qvec

S = math.sqrt(0.5)
IDENTITY = np.eye(3)
RZ90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # x -> y
RX180 = np.diag([1.0, -1.0, -1.0])
CYCLE = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 120 deg about (1, 1, 1): x -> y -> z -> x


def rot_z(degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_cam(R, center, name="cam", fov=0.8, width=800, height=600) -> CameraInfo:
    R = np.asarray(R, dtype=np.float64)
    T = -R.T @ np.asarray(center, dtype=np.float64)
    return CameraInfo(camera_id=0, R=R, T=T, FovY=fov, FovX=fov, image_path=None, image_name=name, width=width, height=height)


def cam_center(cam: CameraInfo) -> np.ndarray:
    return -cam.R @ cam.T


class QuaternionConversionTest(unittest.TestCase):
    def test_qvec2rotmat_converts_a_batch_of_wxyz_quaternions(self):
        qvecs = np.array([[1.0, 0.0, 0.0, 0.0], [S, 0.0, 0.0, S], [0.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
        np.testing.assert_allclose(qvec2rotmat(qvecs), np.stack([IDENTITY, RZ90, RX180, CYCLE]), atol=1e-12)

    def test_qvec2rotmat_normalizes_the_quaternion(self):
        np.testing.assert_allclose(qvec2rotmat(np.array([0.0, 0.0, 0.0, 3.0])), np.diag([-1.0, -1.0, 1.0]), atol=1e-12)

    def test_rotmat2qvec_converts_a_batch_to_wxyz_with_nonnegative_w(self):
        # A -170 deg z rotation is the case where a Shepperd-style conversion yields w < 0; COLMAP stores w >= 0.
        half_angle = math.radians(-85.0)
        expected = np.array([[1.0, 0.0, 0.0, 0.0], [S, 0.0, 0.0, S], [0.5, 0.5, 0.5, 0.5], [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]])
        np.testing.assert_allclose(rotmat2qvec(np.stack([IDENTITY, RZ90, CYCLE, rot_z(-170.0)])), expected, atol=1e-12)

    def test_rotmat2qvec_projects_non_orthonormal_input_to_the_nearest_rotation(self):
        np.testing.assert_allclose(rotmat2qvec(2.0 * RZ90), [S, 0.0, 0.0, S], atol=1e-12)

    def test_rotmat2qvec_handles_half_turns(self):
        axes = np.random.default_rng(0).normal(size=(32, 3))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        half_turns = 2 * axes[:, :, None] * axes[:, None, :] - np.eye(3)  # 180 deg about each axis
        qvecs = rotmat2qvec(half_turns)
        np.testing.assert_allclose(np.linalg.norm(qvecs, axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(qvec2rotmat(qvecs), half_turns, atol=1e-12)

    def test_conversions_round_trip_random_quaternions(self):
        qvecs = np.random.default_rng(0).normal(size=(256, 4))
        qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True)
        qvecs[qvecs[:, 0] < 0] *= -1
        rotmats = qvec2rotmat(qvecs)
        np.testing.assert_allclose(rotmats @ rotmats.transpose(0, 2, 1), np.broadcast_to(np.eye(3), rotmats.shape), atol=1e-12)
        np.testing.assert_allclose(rotmat2qvec(rotmats), qvecs, atol=1e-12)


class ColmapCameraIOTest(unittest.TestCase):
    def setUp(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.root = Path(tmp_dir.name)

    def _read(self) -> list[CameraInfo]:
        return readColmapCameras(str(self.root / "images.txt"), str(self.root / "cameras.txt"), "images")

    def test_readColmapCameras_returns_camera_to_world_rotation(self):
        (self.root / "cameras.txt").write_text("1 PINHOLE 800 600 400 400 400 300\n")
        (self.root / "images.txt").write_text(f"1 1 0 0 0 0.5 0 0 1 a.png\n\n2 {S} 0 0 {S} 1 2 3 1 b.png\n\n")

        cams = self._read()

        self.assertEqual([cam.image_name for cam in cams], ["a", "b"])
        np.testing.assert_allclose(cams[0].R, IDENTITY, atol=1e-12)
        np.testing.assert_allclose(cams[1].R, RZ90.T, atol=1e-12)  # COLMAP stores the world-to-camera rotation
        np.testing.assert_allclose(cams[1].T, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(cams[1].FovX, 2 * math.atan(1.0))
        self.assertAlmostEqual(cams[1].FovY, 2 * math.atan(0.75))

    def test_readColmapCameras_accepts_a_model_without_images(self):
        (self.root / "cameras.txt").write_text("1 PINHOLE 800 600 400 400 400 300\n")
        (self.root / "images.txt").write_text("# no registered images\n")
        self.assertEqual(self._read(), [])

    def test_camInfosToColmap_round_trips_through_readColmapCameras(self):
        rng = np.random.default_rng(1)
        fov_y, fov_x = 2 * math.atan(300 / 600), 2 * math.atan(400 / 600)  # PINHOLE 800x600 with f = 600
        cam_infos = [
            CameraInfo(camera_id=1, R=R, T=rng.normal(size=3), FovY=fov_y, FovX=fov_x, image_path=None, image_name=f"img{i}", width=800, height=600)
            for i, R in enumerate(Rotation.random(5, random_state=1).as_matrix())
        ]

        camInfosToColmap(cam_infos, str(self.root))
        cams = self._read()

        self.assertEqual([cam.image_name for cam in cams], [cam.image_name for cam in cam_infos])
        for cam, ref in zip(cams, cam_infos):
            np.testing.assert_allclose(cam.R, ref.R, atol=1e-12)
            np.testing.assert_allclose(cam.T, ref.T, atol=1e-12)
            self.assertAlmostEqual(cam.FovX, fov_x)
            self.assertAlmostEqual(cam.FovY, fov_y)

    def test_camInfosToColmap_writes_an_empty_model_for_no_cameras(self):
        camInfosToColmap([], str(self.root))
        self.assertEqual((self.root / "images.txt").read_text(), "")
        self.assertEqual((self.root / "cameras.txt").read_text(), "")


class InterpolateCameraInfosTest(unittest.TestCase):
    def test_slerps_rotation_and_lerps_position_with_sqrt_easing(self):
        cams = interpolateCameraInfos([make_cam(IDENTITY, [0, 0, 0])], [make_cam(RZ90, [2, 0, 0])], num_interp=4)

        self.assertEqual(len(cams), 4)
        for cam, alpha in zip(cams, np.sqrt([0.25, 0.5, 0.75, 1.0])):
            np.testing.assert_allclose(cam.R, rot_z(90.0 * alpha), atol=1e-12)
            np.testing.assert_allclose(cam_center(cam), [2.0 * alpha, 0.0, 0.0], atol=1e-12)

    def test_takes_the_shortest_arc(self):
        cams = interpolateCameraInfos([make_cam(rot_z(170.0), [0, 0, 0])], [make_cam(rot_z(-170.0), [0, 0, 0])], num_interp=4)
        np.testing.assert_allclose(cams[0].R, rot_z(180.0), atol=1e-12)  # alpha = 0.5 of the 20 deg arc through 180 deg

    def test_interpolates_from_the_closest_source_in_step_major_order(self):
        sources = [make_cam(IDENTITY, [0, 0, 0]), make_cam(IDENTITY, [100, 0, 0])]
        targets = [make_cam(RZ90, [101, 0, 0], fov=0.5, width=640), make_cam(RZ90, [1, 0, 0], fov=0.6, width=320)]

        cams = interpolateCameraInfos(sources, targets, num_interp=4)

        self.assertEqual([cam.image_name for cam in cams[:3]], ["interp_00_0000", "interp_00_0001", "interp_01_0000"])
        self.assertEqual(len(cams), 8)
        np.testing.assert_allclose(cam_center(cams[0]), [100.5, 0.0, 0.0], atol=1e-12)  # alpha 0.5 from source 1
        np.testing.assert_allclose(cam_center(cams[1]), [0.5, 0.0, 0.0], atol=1e-12)  # alpha 0.5 from source 0
        self.assertEqual((cams[1].FovX, cams[1].FovY, cams[1].width, cams[1].height), (0.6, 0.6, 320, 600))


if __name__ == "__main__":
    unittest.main()
