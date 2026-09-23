import math
import unittest

import numpy as np
import torch

from src.diff_recon.trainers.TS_trainer import TSTrainer
from src.diff_recon.trainers.trainer_utils import LPIPS, ssimLoss
from src.diff_recon.utils.camera import Camera
from src.diff_recon.utils.config import Config


class Views:
    def __init__(self, cameras):
        self.cameras = cameras

    def getTestDataset(self):
        return iter(self.cameras)


class FixedRenders:
    """Model stand-in whose render for each view is a fixed perturbation of its ground truth."""

    def __init__(self, renders):
        self.renders = renders
        self.get_vertex = torch.zeros(1, 3, 3)

    def forward(self, camera, background, is_training, color_affine):
        return {"render": self.renders[camera.uid].to(camera.device)}


class ScalarLog:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, step):
        self.scalars[tag] = value

    def add_image(self, *args):
        pass

    def info(self, message):
        pass

    def debug(self, message):
        pass


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class EvaluateTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        images = [rng.random((3, 64, 64), dtype=np.float32) for _ in range(4)]
        cameras = [Camera(R=np.eye(3), T=np.zeros(3), FoVx=math.pi / 2, gt_image=image, uid=i) for i, image in enumerate(images)]
        renders = [torch.from_numpy(np.clip(image + rng.normal(0, 0.1, image.shape).astype(np.float32), 0, 1)) for image in images]

        self.trainer = trainer = TSTrainer.__new__(TSTrainer)
        trainer.config = Config(trainer=Config())
        trainer.device = torch.device("cuda")
        trainer.dataset = Views(cameras)
        trainer.model = FixedRenders(renders)
        trainer.logger = ScalarLog()
        trainer.ssimLoss = ssimLoss.to(trainer.device)
        trainer.lpips = LPIPS(net_type="vgg", reduction="mean", normalize=True).to(trainer.device)
        trainer._save_img_idx = []
        trainer._tb_gt_recorded = False
        self.renders, self.cameras = renders, cameras

    def test_repeated_evaluations_do_not_accumulate_lpips_scores(self):
        # LPIPS.forward clones every accumulated score, so any retained state makes each later call slower
        for iteration in range(3):
            self.trainer._evaluate(iteration)
        self.assertEqual(len(self.trainer.lpips.all_scores), 0)

    def test_evaluation_reports_the_mean_of_independent_per_view_lpips(self):
        reference = LPIPS(net_type="vgg", reduction="mean", normalize=True).to(self.trainer.device)
        with torch.no_grad():
            expected = np.mean([reference(render.cuda()[None], camera.gt_image.cuda()[None]).item() for render, camera in zip(self.renders, self.cameras)])
        for iteration in range(2):
            self.trainer._evaluate(iteration)
            self.assertAlmostEqual(self.trainer.logger.scalars["Average LPIPS"], expected, places=6)


if __name__ == "__main__":
    unittest.main()
