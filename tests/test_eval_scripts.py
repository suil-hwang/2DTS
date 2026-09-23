import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))  # eval_tnt imports its sibling `registration`
import eval_tnt  # noqa: E402
from scripts.eval_nerf_synthetic import _mean_squared_point_distance  # noqa: E402


def sample_ellipsoid(rng: np.random.Generator, n: int) -> np.ndarray:
    directions = rng.normal(size=(n, 3))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True) * [0.6, 0.4, 0.3]


def write_point_cloud(path: Path, points: np.ndarray) -> None:
    o3d.io.write_point_cloud(str(path), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)))


def write_trajectory(path: Path, centers: np.ndarray) -> None:
    with open(path, "w") as f:
        for i, center in enumerate(centers):
            pose = np.eye(4)
            pose[:3, 3] = center
            f.write(f"{i} {i} 0\n" + "".join(" ".join(map(repr, row.tolist())) + "\n" for row in pose))


class ChamferDistanceTest(unittest.TestCase):
    def test_mean_squared_point_distance_matches_brute_force(self):
        rng = np.random.default_rng(0)
        query = rng.normal(size=(300, 3)).astype(np.float32)
        reference = rng.normal(size=(200, 3)).astype(np.float32)
        q, r = query.astype(np.float64), reference.astype(np.float64)
        expected = ((q[:, None] - r[None]) ** 2).sum(axis=-1).min(axis=1).mean()

        result = _mean_squared_point_distance(query, reference)

        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, expected, delta=1e-12 * expected)

    def test_mean_squared_point_distance_is_zero_for_a_superset_reference(self):
        points = np.random.default_rng(1).random((50, 3)).astype(np.float32)
        self.assertEqual(_mean_squared_point_distance(points[:20], points), 0.0)


class FScoreTest(unittest.TestCase):
    def test_counts_distances_strictly_below_the_threshold(self):
        precision_distances = o3d.utility.DoubleVector([0.1, 0.4, 0.5, 0.9])
        recall_distances = o3d.utility.DoubleVector([0.2, 0.6, 0.7])

        precision, recall, fscore, edges_source, cum_source, *_ = eval_tnt.get_f1_score_histo2(0.5, None, 5, precision_distances, recall_distances)

        self.assertEqual((precision, recall), (0.5, 1 / 3))
        self.assertAlmostEqual(fscore, 2 * 0.5 * (1 / 3) / (0.5 + 1 / 3))
        self.assertEqual((type(precision), type(recall)), (float, float))
        self.assertEqual(len(edges_source), len(cum_source) + 1)
        self.assertEqual(cum_source[-1], 1.0)

    def test_returns_zero_scores_without_distances(self):
        precision, recall, fscore, *_ = eval_tnt.get_f1_score_histo2(0.5, None, 5, o3d.utility.DoubleVector(), o3d.utility.DoubleVector([0.1]))
        self.assertEqual((precision, recall, fscore), (0, 0, 0))


class TanksAndTemplesEvaluationTest(unittest.TestCase):
    def test_aligns_a_reconstruction_given_in_the_colmap_frame(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        scene = "Courthouse"  # tau = 0.025
        dataset_dir, out_dir = Path(tmp_dir.name) / scene, Path(tmp_dir.name) / "evaluation"
        dataset_dir.mkdir()

        # The reconstruction lives in the COLMAP frame; the ground truth is a scaled, rotated and shifted copy of it.
        rng = np.random.default_rng(0)
        gt_trans = np.eye(4)
        gt_trans[:3, :3] = 1.7 * Rotation.from_euler("xyz", [20, -30, 45], degrees=True).as_matrix()
        gt_trans[:3, 3] = [0.3, -0.2, 1.0]
        gt_points = sample_ellipsoid(rng, 200_000) @ gt_trans[:3, :3].T + gt_trans[:3, 3]
        angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        centers = np.column_stack([1.5 * np.cos(angles), 1.5 * np.sin(angles), 0.5 * np.sin(3 * angles)])

        write_point_cloud(dataset_dir / f"{scene}.ply", gt_points)
        write_point_cloud(Path(tmp_dir.name) / "recon.ply", sample_ellipsoid(rng, 200_000))
        write_trajectory(dataset_dir / f"{scene}_COLMAP_SfM.log", centers)
        write_trajectory(Path(tmp_dir.name) / "traj.log", centers)
        np.savetxt(dataset_dir / f"{scene}_trans.txt", gt_trans)
        lo, hi = gt_points.min(axis=0) - 0.2, gt_points.max(axis=0) + 0.2
        crop = {
            "class_name": "SelectionPolygonVolume",
            "version_major": 1,
            "version_minor": 0,
            "orthogonal_axis": "Z",
            "axis_min": lo[2],
            "axis_max": hi[2],
            "bounding_polygon": [[lo[0], lo[1], 0.0], [hi[0], lo[1], 0.0], [hi[0], hi[1], 0.0], [lo[0], hi[1], 0.0]],
        }
        (dataset_dir / f"{scene}.json").write_text(json.dumps(crop))

        eval_tnt.run_evaluation(str(dataset_dir), str(Path(tmp_dir.name) / "traj.log"), str(Path(tmp_dir.name) / "recon.ply"), str(out_dir))

        with open(out_dir / "result.csv", newline="") as f:
            header, values = list(csv.reader(f))
        self.assertEqual(header, ["precision", "recall", "fscore"])
        for name, value in zip(header, map(float, values)):
            self.assertGreater(value, 0.99, name)


if __name__ == "__main__":
    unittest.main()
