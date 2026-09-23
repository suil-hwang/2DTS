import unittest

import numpy as np

from src.diff_recon.models.raw_triangle import RawTriangle


def make_triangles(vertex: np.ndarray) -> RawTriangle:
    n = len(vertex)
    return RawTriangle(vertex.astype(np.float32), np.arange(n, dtype=np.float32)[:, None], np.zeros((n, 3), dtype=np.float32))


class SubtractTest(unittest.TestCase):
    def test_removes_triangles_whose_centers_coincide(self):
        vertex = np.random.default_rng(0).random((6, 3, 3))
        triangles = make_triangles(vertex)

        triangles -= make_triangles(vertex[[4, 1]] + [0.0, 0.0, 5e-6])  # centers 5e-6 apart still count as the same triangle

        np.testing.assert_array_equal(triangles.opacity[:, 0], [0, 2, 3, 5])
        np.testing.assert_array_equal(triangles.vertex, vertex[[0, 2, 3, 5]].astype(np.float32))
        self.assertTrue(triangles.contained_idx.all())

    def test_keeps_triangles_farther_than_the_tolerance(self):
        vertex = np.random.default_rng(1).random((3, 3, 3))
        triangles = make_triangles(vertex)

        difference = triangles - make_triangles(vertex + [0.0, 1e-4, 0.0])

        self.assertEqual(len(difference), 3)
        self.assertEqual(len(triangles), 3)


if __name__ == "__main__":
    unittest.main()
