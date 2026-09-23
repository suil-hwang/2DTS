import math
import time
import unittest

import numpy as np
import torch

from src.diff_recon.utils.camera import Camera

RX90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # camera +z (forward) -> world -y


def rot_z(degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_camera(R=np.eye(3), center=(0.0, 0.0, 0.0), width=4, height=2, fov_x=math.pi / 2, fov_y=math.pi / 2, **kwargs) -> Camera:
    R = np.asarray(R, dtype=np.float64)
    T = -R.T @ np.asarray(center, dtype=np.float64)
    return Camera(R=R, T=T, FoVx=fov_x, FoVy=fov_y, image_width=width, image_height=height, **kwargs)


class CameraGeometryTest(unittest.TestCase):
    def test_project_points_maps_known_points(self):
        camera = make_camera(znear=1.0, zfar=100.0)  # 90 deg FoV: x / z = 1 lands on the image border
        points = torch.tensor([[0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [0.0, -1.0, 2.0]])

        np.testing.assert_allclose(camera.project_points(points, "view"), points.numpy(), atol=1e-6)
        ndc_z = 100.0 * (2.0 - 1.0) / (2.0 * (100.0 - 1.0))
        np.testing.assert_allclose(camera.project_points(points, "ndc"), [[0.0, 0.0, ndc_z], [1.0, 0.0, ndc_z], [0.0, -0.5, ndc_z]], atol=1e-6)
        np.testing.assert_allclose(camera.project_points(points, "screen"), [[2.0, 1.0], [4.0, 1.0], [2.0, 0.5]], atol=1e-6)
        with self.assertRaises(ValueError):
            camera.project_points(points, "clip")

    def test_project_points_view_applies_the_world_to_camera_pose(self):
        center = np.array([1.0, 2.0, 3.0])
        camera = make_camera(R=RX90, center=center)
        world_point = center + RX90 @ np.array([0.5, -0.25, 2.0])
        np.testing.assert_allclose(camera.project_points(torch.tensor(world_point[None], dtype=torch.float32), "view"), [[0.5, -0.25, 2.0]], atol=1e-6)

    def test_camera_center_is_the_float32_rounded_camera_position(self):
        center = np.array([5000.25, -3000.5, 200.125])  # exactly representable in float32
        camera = make_camera(R=rot_z(30.0), center=center)
        self.assertTrue(torch.equal(camera.camera_center, torch.tensor(center, dtype=torch.float32)))

    def test_fovy_is_derived_from_the_image_size_without_an_image(self):
        camera = Camera(R=np.eye(3), T=np.zeros(3), FoVx=0.8, image_width=800, image_height=400)
        self.assertAlmostEqual(camera.FoVy, 2 * math.atan(math.tan(0.4) * 0.5))
        self.assertAlmostEqual(camera.tan_fovy, math.tan(0.4) * 0.5)

    def test_get_rays_pass_through_pixel_centers_like_the_rasterizer(self):
        camera = make_camera(R=RX90, width=3, height=2, fov_x=1.0, fov_y=0.6)
        # rasterizer: pixToProj(v, S) = (2v - S + 1) / S, ray = (tan_fovx * pixToProj(x), tan_fovy * pixToProj(y), 1)
        xs = [(2 * x - 3 + 1) / 3 * math.tan(0.5) for x in range(3)]
        ys = [(2 * y - 2 + 1) / 2 * math.tan(0.3) for y in range(2)]
        expected = np.array([[[x, y, 1.0] for x in xs] for y in ys])

        np.testing.assert_allclose(camera.get_rays(), expected, atol=1e-6)
        np.testing.assert_allclose(camera.get_rays(world_space=True), expected @ RX90.T, atol=1e-6)

    def test_get_xyz_from_depth_round_trips_through_project_points(self):
        camera = make_camera(R=rot_z(30.0) @ RX90, center=(1.0, 2.0, 3.0), width=40, height=30, fov_x=1.2, fov_y=0.9)
        depth = torch.rand(30, 40, generator=torch.Generator().manual_seed(0)) * 5 + 1

        xyz = camera.get_xyz_from_depth(depth).view(-1, 3)

        np.testing.assert_allclose(camera.project_points(xyz, "view")[:, 2], depth.view(-1), atol=1e-5)
        np.testing.assert_allclose(camera.project_points(xyz, "screen"), camera.get_pixels().view(-1, 2), atol=1e-4)

    def test_gt_image_is_clamped_without_aliasing_the_source_array(self):
        image = np.full((3, 2, 2), 0.5, dtype=np.float32)
        image[0, 0, 0] = 1.5
        camera = Camera(R=np.eye(3), T=np.zeros(3), FoVx=0.8, gt_image=image)
        image[:] = 0.25
        self.assertEqual(camera.gt_image[0, 0, 0].item(), 1.0)
        self.assertEqual(camera.gt_image[1, 1, 1].item(), 0.5)


def host_time_while_gpu_busy(fn) -> tuple[float, float]:
    """Return (host time of fn, GPU-busy window) with ~100 ms of queued GPU work; a host sync makes them equal."""
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.cuda._sleep(200_000_000)
    queued = time.perf_counter()
    fn()
    returned = time.perf_counter()
    torch.cuda.synchronize()
    return returned - queued, time.perf_counter() - start


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class CameraDeviceTest(unittest.TestCase):
    def setUp(self):
        self.camera = make_camera(R=RX90, center=(1.0, 2.0, 3.0), width=64, height=48, bg_color=np.ones(3)).to("cuda")

    def assert_does_not_block_host(self, fn):
        host_time, busy_time = host_time_while_gpu_busy(fn)
        self.assertLess(host_time, 0.25 * busy_time, f"host blocked {host_time * 1e3:.1f} ms of a {busy_time * 1e3:.1f} ms GPU-busy window")

    def test_to_moves_every_tensor_and_the_neighbor_camera(self):
        camera = make_camera(bg_color=np.ones(3), gt_image=np.zeros((3, 2, 4), dtype=np.float32), gt_alpha_mask=np.ones((2, 4), dtype=np.float32))
        camera.neighbor_cam = make_camera()
        camera.pin_memory().to("cuda")
        for tensor in (camera.gt_image, camera.alpha_mask, camera.world_view_transform, camera.projection_matrix, camera.full_proj_transform, camera.camera_center, camera.bg_color, camera.neighbor_cam.world_view_transform):
            self.assertEqual(tensor.device.type, "cuda")

    def test_get_xyz_from_depth_does_not_block_host(self):
        depth = torch.ones(48, 64, device="cuda")
        self.assert_does_not_block_host(lambda: self.camera.get_xyz_from_depth(depth))

    def test_project_points_does_not_block_host(self):
        points = torch.rand(1000, 3, device="cuda")
        for target in ("ndc", "screen", "view"):
            self.assert_does_not_block_host(lambda: self.camera.project_points(points, target))


if __name__ == "__main__":
    unittest.main()
