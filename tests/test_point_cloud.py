import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData

from src.diff_recon.models.point_cloud import PointCloud


class StorePlyTest(unittest.TestCase):
    def setUp(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.path = str(Path(tmp_dir.name) / "nested" / "points.ply")

        self.points = np.array([[0.5, -1.25, 3.0], [1e-3, 2.0, -4.5], [7.0, 8.0, 9.0]])
        self.normals = np.array([[0.0, 0.0, 1.0], [0.6, 0.8, 0.0], [0.0, -1.0, 0.0]])
        self.colors = np.array([[0.0, 0.5, 1.0], [0.999, 0.25, 0.1], [0.2, 0.4, 0.6]])

    def test_writes_float32_geometry_and_truncated_uint8_colors(self):
        PointCloud(self.points, self.colors, self.normals).storePly(self.path)

        vertices = PlyData.read(self.path)["vertex"].data
        self.assertEqual(vertices.dtype.names, ("x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"))
        np.testing.assert_array_equal(np.column_stack([vertices[name] for name in ("x", "y", "z")]), self.points.astype(np.float32))
        np.testing.assert_array_equal(np.column_stack([vertices[name] for name in ("nx", "ny", "nz")]), self.normals.astype(np.float32))
        rgb = np.column_stack([vertices[name] for name in ("red", "green", "blue")])
        self.assertEqual(rgb.dtype, np.uint8)
        np.testing.assert_array_equal(rgb, [[0, 127, 255], [254, 63, 25], [51, 102, 153]])

    def test_round_trips_through_fetchPly(self):
        PointCloud(self.points.astype(np.float32), self.colors.astype(np.float32), self.normals.astype(np.float32)).storePly(self.path)

        pcd = PointCloud().fetchPly(self.path)

        np.testing.assert_array_equal(pcd.points, self.points.astype(np.float32))
        np.testing.assert_array_equal(pcd.normals, self.normals.astype(np.float32))
        np.testing.assert_allclose(pcd.colors, np.floor(self.colors.astype(np.float32) * 255) / 255)

    def test_writes_an_empty_cloud(self):
        empty = np.empty((0, 3))
        PointCloud(empty, empty, empty).storePly(self.path)
        self.assertEqual(len(PointCloud().fetchPly(self.path)), 0)


if __name__ == "__main__":
    unittest.main()
