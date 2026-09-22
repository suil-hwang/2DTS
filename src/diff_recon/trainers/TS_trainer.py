import os
import time

import numpy as np
import torch
from tqdm import tqdm

from .trainer_utils import *
from ..datasets.Base_dataset import BaseDatasetFactory
from ..models.TS_model import TSModel
from ..models.point_cloud import PointCloud
from ..models.model_utils import grid_sampling, to_numpy
from ..utils.timer import Timer
from ..utils.config import Config, loadConfig
from ..utils.logger import Logger
from ..utils.camera import Camera
from ..utils.vis_utils import save_image_tensor


class TSTrainer:
    """Coordinate 2DTS initialization, optimization, evaluation, and exports."""

    def __init__(self, config: str | Config = None, exp_name: str = None, device: torch.device | int = None, log_file: bool = True) -> None:
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        default_config = Config(trainer=Config(), dataset=Config(), model=Config())
        self.config = config if isinstance(config, Config) else loadConfig(config) if config is not None else default_config
        self.exp_name = time_str if not exp_name else exp_name
        self.device = device if isinstance(device, torch.device) else torch.device(f"cuda:{device}") if device is not None else torch.device("cuda")

        config = self.config.trainer
        output_dir = config.output_dir if config.output_dir is not None else ".output"
        clean_output_dir = config.clean_output_dir if config.clean_output_dir is not None else False
        seed = config.seed if config.seed is not None else 42
        detect_anomaly = config.detect_anomaly if config.detect_anomaly is not None else False
        use_tensorboard = config.use_tensorboard if config.use_tensorboard is not None else True

        self.output_dir = os.path.join(output_dir, self.exp_name)
        if clean_output_dir and log_file:
            os.system(f"rm -rf {self.output_dir}")

        self.logger = Logger(time_str, os.path.join(self.output_dir, "log") if log_file else None, use_tensorboard=use_tensorboard)
        self.logger.info(f"config args: {self.config}")
        if not log_file:
            self.logger.warning("Not creating log file because log_file is set to False")

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.autograd.set_detect_anomaly(detect_anomaly)

        self.dataset = self._load_dataset()

        # Initialize model
        train_size = self.dataset.getTrainDatasetSize()
        pruning = getattr(self.config.model.model_update, "contribution_pruning", None)
        if pruning is not None and pruning.interval_iter == -1:
            assert pruning.start_iter % train_size == 0, "Auto pruning interval requires start_iter to align with train_size"
            pruning.interval_iter = train_size
        self.model = TSModel(self.config.model, logger=self.logger, device=self.device)
        self.model.setup_color_affine(train_size)
        self.model.setup_scene_info(self.dataset.getSceneInfo())

        # Initialize loss module
        self.ssimLoss = ssimLoss.to(self.device)
        self.lpips = LPIPS(net_type="vgg", reduction="mean", normalize=True).to(self.device)
        if self.config.trainer.w_dog > 0:
            self.dogLoss = dogLoss.to(self.device)
        if self.config.trainer.smoothness_loss:
            self.smoothnessLoss = smoothnessLoss.to(self.device)
        if self.config.trainer.geometry_loss and self.config.trainer.geometry_loss.w_geometry > 0:
            scale_factor = self.config.trainer.geometry_loss.scale_factor if self.config.trainer.geometry_loss.scale_factor is not None else 1.0
            depth_grad = self.config.trainer.geometry_loss.depth_grad if self.config.trainer.geometry_loss.depth_grad is not None else False
            self.depthNormalLoss = DepthNormalLoss(depth_grad=depth_grad, scale_factor=scale_factor)
        if self.config.trainer.consistency_loss and (self.config.trainer.consistency_loss.w_geo > 0 or self.config.trainer.consistency_loss.w_color > 0):
            error_thres = self.config.trainer.consistency_loss.error_thres
            n_sample = self.config.trainer.consistency_loss.n_sample
            patch_size = self.config.trainer.consistency_loss.patch_size
            dilation = self.config.trainer.consistency_loss.dilation
            self.consistencyLoss = ConsistencyLoss(error_thres=error_thres, n_sample=n_sample, patch_size=patch_size, dilation=dilation)
        self._nearest_indices_cache = None
        self._nearest_vertex = None

        self.accum_loss = []

        # For logging and profiling
        test_img_count = self.dataset.getTestDatasetSize()
        eval_save_img_count = self.config.trainer.eval_save_img_count if self.config.trainer.eval_save_img_count is not None else 3
        if test_img_count < eval_save_img_count:
            self._save_img_idx = list(range(test_img_count))
        else:
            self._save_img_idx = sorted(np.random.choice(test_img_count, eval_save_img_count, replace=False).tolist())
        self._tb_gt_recorded = False

    def _load_dataset(self) -> BaseDatasetFactory:
        dataset_type = self.config.dataset.type
        match dataset_type:
            case "Colmap":
                from ..datasets.Colmap_dataset import ColmapDatasetFactory

                return ColmapDatasetFactory(self.config.dataset, self.logger)
            case "NerfSynthetic":
                from ..datasets.NerfSynthetic_dataset import NerfSyntheticDatasetFactory

                return NerfSyntheticDatasetFactory(self.config.dataset, self.logger)
            case "MatrixCity":
                from ..datasets.MatrixCity_dataset import MatrixCityDatasetFactory

                return MatrixCityDatasetFactory(self.config.dataset, self.logger)
            case _:
                raise ValueError(f"Unknown dataset type: {dataset_type}")

    def _get_loss(self, iteration: int, render_pkg: dict) -> None:
        def scheduled_weight(section, name):
            if section is not None and iteration > section.start_iter:
                return getattr(section, name)
            return 0

        cam: Camera = render_pkg["camera"]
        gt_image = cam.gt_image
        gt_mask = cam.alpha_mask
        image = render_pkg["render"]
        scaling = render_pkg["scaling"]
        opacity = render_pkg["opacity"]
        vertex = render_pkg["vertex"]
        depth = render_pkg["depth"]
        normal = render_pkg["normal"]
        distortion = render_pkg["distortion"]
        alpha_mask = render_pkg["alpha_mask"]

        # retrieve parameters
        config = self.config.trainer
        w_L1 = config.w_L1 if config.w_L1 is not None else 1.0
        w_L2 = config.w_L2 if config.w_L2 is not None else 0
        w_ssim = config.w_ssim if config.w_ssim is not None else 0
        w_dog = config.w_dog if config.w_dog is not None else 0
        w_geometry = scheduled_weight(config.geometry_loss, "w_geometry")
        w_distortion = scheduled_weight(config.distortion_loss, "w_distortion")

        w_smoothness_image = scheduled_weight(config.smoothness_loss, "w_image")
        w_smoothness_normal = scheduled_weight(config.smoothness_loss, "w_normal")
        w_smoothness_depth = scheduled_weight(config.smoothness_loss, "w_depth")
        smoothness_scales = (
            config.smoothness_loss.scale_factor if (config.smoothness_loss is not None and config.smoothness_loss.scale_factor is not None) else [1.0]
        )

        w_consistency_geo = scheduled_weight(config.consistency_loss, "w_geo")
        w_consistency_color = scheduled_weight(config.consistency_loss, "w_color")

        w_scaling_reg = config.w_scaling_reg if config.w_scaling_reg is not None else 0
        w_affine_reg = config.w_affine_reg if config.w_affine_reg is not None else 0
        w_opacity_reg = scheduled_weight(config.opacity_reg, "w_opacity_reg")
        w_vertex_reg = scheduled_weight(config.vertex_reg, "w_vertex_reg")

        # pixelwise losses
        if gt_mask is not None and config.train_alpha_mask:
            gt_image = gt_image * gt_mask
            image = image * gt_mask
            alpha_mask = alpha_mask * gt_mask
        l1_loss = L1(image, gt_image) if w_L1 > 0 else 0
        l2_loss = L2(image, gt_image) if w_L2 > 0 else 0
        ssim_loss = self.ssimLoss(image, gt_image) if w_ssim > 0 else 0
        dog_loss = self.dogLoss(image, gt_image) if w_dog > 0 else 0
        geometry_loss = self.depthNormalLoss(depth, normal, cam.tan_fovx, cam.tan_fovy, alpha_mask, gt_image) if w_geometry > 0 else 0
        distortion_loss = distortion.mean() / self.dataset.cameras_extent if w_distortion > 0 else 0

        image_smoothness_loss = (
            torch.stack([self.smoothnessLoss(image, gt_image, gt_mask, scale) for scale in smoothness_scales]).mean() if w_smoothness_image > 0 else 0
        )
        normal_smoothness_loss = (
            torch.stack([self.smoothnessLoss(normal, gt_image, gt_mask, scale) for scale in smoothness_scales]).mean() if w_smoothness_normal > 0 else 0
        )
        depth_smoothness_loss = (
            torch.stack([self.smoothnessLoss(depth, gt_image, gt_mask, scale) for scale in smoothness_scales]).mean() if w_smoothness_depth > 0 else 0
        )

        if w_consistency_geo > 0 or w_consistency_color > 0:
            if cam.neighbor_cam is None:
                raise ValueError("Calculating multiview consistency loss requires setting neighbor_cam for each camera.")
            if (cam.gt_image is None or cam.neighbor_cam.gt_image is None) and w_consistency_color > 0:
                raise ValueError("Calculating multiview consistency loss with ncc loss requires ground truth images for both views.")
            render_pkg_ref = self.model.forward(cam.neighbor_cam, config.train_background)
            consistency_geo_loss, consistency_color_loss = self.consistencyLoss(self.model, render_pkg, render_pkg_ref)
        else:
            consistency_geo_loss, consistency_color_loss = 0, 0

        # regularization losses
        scaling_reg = scaling.mean()

        opacity_reg = 0
        if w_opacity_reg > 0:
            if config.opacity_reg.type == "linear":
                opacity_reg = (1 - opacity).mean()
            elif config.opacity_reg.type == "quad":
                opacity_reg = (0.25 - (opacity - 0.5) ** 2).mean()
            else:
                raise ValueError(f"Unknown opacity regularization type: {config.opacity_reg.type}")

        affine_reg = 0
        if w_affine_reg > 0 and "render_original" in render_pkg:
            image_original = render_pkg["render_original"]
            if gt_mask is not None and config.train_alpha_mask:
                image_original = image_original * gt_mask
            affine_reg = L1(image, image_original)

        vertex_reg = 0
        if w_vertex_reg > 0:
            if (
                (iteration - 1) % config.vertex_reg.interval_iter == 0
                or self._nearest_indices_cache is None
                or vertex is not self._nearest_vertex
            ):
                self._nearest_indices_cache = nearest_neighbor(vertex.view(-1, 3), 3)
                self._nearest_vertex = vertex
            vertex_reg = nearest_dist2(vertex.view(-1, 3), self._nearest_indices_cache).mean()

        # Combine losses
        img_loss = (
            w_L1 * l1_loss
            + w_L2 * l2_loss
            + w_ssim * ssim_loss
            + w_dog * dog_loss
            + w_geometry * geometry_loss
            + w_distortion * distortion_loss
            + w_smoothness_image * image_smoothness_loss
            + w_smoothness_normal * normal_smoothness_loss
            + w_smoothness_depth * depth_smoothness_loss
            + w_consistency_geo * consistency_geo_loss
            + w_consistency_color * consistency_color_loss
        )
        reg_loss = w_scaling_reg * scaling_reg + w_opacity_reg * opacity_reg + w_affine_reg * affine_reg + w_vertex_reg * vertex_reg
        loss = img_loss + reg_loss

        render_pkg["loss"] = loss
        render_pkg["geometry_loss"] = geometry_loss
        render_pkg["distortion_loss"] = distortion_loss
        render_pkg["vertex_loss"] = vertex_reg

    def _optimize(self, iteration: int, render_pkg: dict):
        bs = self.config.trainer.batch_size if self.config.trainer.batch_size is not None else 1

        render_pkg["grad"] = 0
        self.accum_loss.append(render_pkg["loss"])
        if len(self.accum_loss) < bs:
            return

        total_loss = torch.sum(torch.stack(self.accum_loss))
        total_loss.backward()
        render_pkg["grad"] = render_pkg["grad_holder"].grad

        self.model.update_learning_rate(iteration)
        self.model.optimizer.step()
        self.model.optimizer.zero_grad(set_to_none=True)
        render_pkg["grad_holder"].grad = None  # reset grad holder
        self.accum_loss = []

    def _log_stats(self, log_pkg: dict):
        iteration = log_pkg["iteration"]
        loss = log_pkg["loss"]
        triangle_count = log_pkg["triangle_count"]
        gamma = log_pkg["gamma"]
        sh_degree = log_pkg["sh_degree"]
        time_elapsed = log_pkg["time_elapsed"]
        img = log_pkg["render"]

        torch.cuda.empty_cache()
        allocated_mem, reserved_mem = torch.cuda.memory_allocated(self.device), torch.cuda.memory_reserved(self.device)
        self.logger.info(
            f"[ITER {iteration}] Loss: {loss:.5f}, Triangle Count: {triangle_count}, Gamma: {gamma:.5f}, SH Degree: {sh_degree}, "
            + f"Memory Allocated: {allocated_mem / 2**20:.2f} MB, Reserved: {reserved_mem / 2**20:.2f} MB"
        )

        for tag, name in (
            ("Loss", "loss"),
            ("Geometry Loss", "geometry_loss"),
            ("Distortion Loss", "distortion_loss"),
            ("Vertex Loss", "vertex_loss"),
            ("Triangle Count", "triangle_count"),
        ):
            self.logger.add_scalar(tag, log_pkg[name], iteration)
        self.logger.add_scalar("Training Time (min)", time_elapsed / 60, iteration)

        if self.config.trainer.save_train_img:
            save_image_tensor(img, f"{self.output_dir}/train/{iteration:>05d}.png")

    @torch.no_grad()
    def _histogram(self, iteration: int, render_pkg: dict):
        opacity = render_pkg["opacity"].view(-1)
        scaling = render_pkg["scaling"].view(-1)

        # downsample to reduce tensorboard log size
        sample_num = self.config.trainer.histogram_sample_num if self.config.trainer.histogram_sample_num is not None else 10_000
        opacity = opacity[torch.randperm(opacity.shape[0])[:sample_num]]
        scaling = scaling[torch.randperm(scaling.shape[0])[:sample_num]]

        self.logger.add_histogram("Opacity Histogram", opacity, iteration)
        self.logger.add_histogram("Scaling Histogram", scaling, iteration)

    @torch.no_grad()
    def _evaluate(self, iteration: int, use_tensorboard: bool = True, save_img: bool = False) -> float:
        self.logger.debug("Evaluation started")

        background = self.config.trainer.eval_background
        eval_alpha_mask = self.config.trainer.eval_alpha_mask if self.config.trainer.eval_alpha_mask is not None else True

        psnr_vals = []
        ssim_vals = []
        lpips_vals = []
        for i, test_data in enumerate(self.dataset.getTestDataset()):
            camera = test_data.to(self.device)
            image = self.model.forward(camera, background, False, False)["render"].clip(min=0, max=1)
            gt_image = camera.gt_image
            psnr_vals.append(psnr(image, gt_image, camera.alpha_mask if eval_alpha_mask else None))
            ssim_vals.append(self.ssimLoss(image, gt_image))
            lpips_vals.append(self.lpips(image.unsqueeze(0), gt_image.unsqueeze(0)))

            if use_tensorboard and i in self._save_img_idx:
                img_log_idx = self._save_img_idx.index(i)
                self.logger.add_image(f"Pred {img_log_idx}", image, iteration)
                if not self._tb_gt_recorded:
                    self.logger.add_image(f"GT {img_log_idx}", gt_image, 0)

            if save_img:
                save_image_tensor(image, f"{self.output_dir}/eval/{i:>05d}.png")
                save_image_tensor(gt_image, f"{self.output_dir}/eval_gt/{i:>05d}.png")

        if psnr_vals:
            metric_vals = torch.stack([torch.stack(values) for values in (psnr_vals, ssim_vals, lpips_vals)])
            metric_vals = metric_vals.cpu().numpy().astype(np.float64)
            psnr_vals, ssim_vals, lpips_vals = metric_vals[0], 1.0 - metric_vals[1], metric_vals[2]

        mean_psnr = np.mean(psnr_vals)
        mean_ssim = np.mean(ssim_vals)
        mean_lpips = np.mean(lpips_vals)
        if use_tensorboard:
            self._tb_gt_recorded = True
            self.logger.add_scalar("Average PSNR", mean_psnr, iteration)
            self.logger.add_scalar("Average SSIM", mean_ssim, iteration)
            self.logger.add_scalar("Average LPIPS", mean_lpips, iteration)

        self.logger.info(
            f"[ITER {iteration}] Evaluation PSNR: {mean_psnr:.3f}, SSIM: {mean_ssim:.3f}, LPIPS: {mean_lpips:.3f}, "
            + f"eval view count: {len(psnr_vals)}, triangle count: {self.model.get_vertex.shape[0]}"
        )

        self.logger.debug("Evaluation finished")
        return mean_psnr

    def _train(self):
        dataset: BaseDatasetFactory = self.dataset
        config = self.config.trainer

        # Initialize model
        first_iter = 0
        if config.start_checkpoint:
            self.logger.info(f"Initializing Triangles from checkpoint {config.start_checkpoint}.ckpt")
            self.model.load_ckpt(f"{self.output_dir}/ckpt/{config.start_checkpoint}.ckpt")
            first_iter = config.start_checkpoint
        elif config.start_pointcloud:
            self.logger.info(f"Initializing Triangles from ply {config.start_pointcloud}.ply")
            self.model.loadPLY(f"{self.output_dir}/point_cloud/{config.start_pointcloud}.ply")
            first_iter = config.start_pointcloud
        if self.model.initialized and config.start_opacity is not None:
            self.logger.info(f"Setting initial opacity to {config.start_opacity}")
            self.model.set_opacity(config.start_opacity)
        if not self.model.initialized:
            self.logger.info("Initializing Triangles from point cloud")
            self.model.create_from_pcd(dataset.getPointCloud())
        self.model.state_update(first_iter)

        if config.initial_eval:
            self._evaluate(first_iter, save_img=config.save_eval_img)

        # Start training
        self.logger.info("Training started")
        timer = Timer("Training")
        progress_bar = tqdm(range(first_iter + 1, config.iterations + 1), desc="Training")

        for iteration in progress_bar:
            timer.log("data loading")
            camera = dataset.nextTrainData().to(self.device)

            timer.log("forward pass")
            render_pkg = self.model.forward(camera, config.train_background)

            timer.log("loss calculation")
            self._get_loss(iteration, render_pkg)

            timer.log("optimizer step")
            self._optimize(iteration, render_pkg)

            if config.log_interval_iter > 0 and iteration % config.log_interval_iter == 0:
                timer.log("logging")
                log_pkg = {
                    **render_pkg,
                    "iteration": iteration,
                    "triangle_count": (
                        render_pkg["opacity"].shape[0] if self.model.ste_threshold is None else (render_pkg["opacity"] > self.model.ste_threshold).sum().item()
                    ),
                    "gamma": self.model.gamma,
                    "sh_degree": self.model.active_sh_degree,
                    "time_elapsed": timer.total_duration(),
                }
                self._log_stats(log_pkg)

            if config.histogram_interval_iter > 0 and iteration % config.histogram_interval_iter == 0:
                timer.log("histogram")
                self._histogram(iteration, render_pkg)

            if config.eval_interval_iter > 0 and iteration % config.eval_interval_iter == 0:
                timer.log("evaluation")
                self._evaluate(iteration, save_img=config.save_eval_img)

            timer.log("model update")
            if self.config.trainer.save_train_img:
                render_pkg["densification_img"] = None
            self.model.model_update(iteration, render_pkg)
            if "densification_img" in render_pkg and render_pkg["densification_img"] is not None:
                save_image_tensor(render_pkg["densification_img"], f"{self.output_dir}/train/{iteration:>05d}_densification.png")

            if (config.save_iterations is not None and iteration in config.save_iterations) or (
                config.save_interval_iter > 0 and iteration % config.save_interval_iter == 0
            ):
                timer.log("point cloud saving")
                self.model.savePLY(f"{self.output_dir}/point_cloud/{iteration}.ply")

            if (config.checkpoint_iterations is not None and iteration in config.checkpoint_iterations) or (
                config.ckpt_interval_iter > 0 and iteration % config.ckpt_interval_iter == 0
            ):
                timer.log("checkpoint saving")
                self.model.save_ckpt(f"{self.output_dir}/ckpt/{iteration}.ckpt")

            if config.save_mesh_iterations is not None and iteration in config.save_mesh_iterations:
                timer.log("mesh saving")
                self.model.saveGLB(f"{self.output_dir}/glb/{iteration}.glb")
                self.model.saveGLB(f"{self.output_dir}/mesh_ply/{iteration}_mesh.ply")

            if config.save_pcd_iterations is not None and iteration in config.save_pcd_iterations:
                timer.log("pcd saving")
                n_sample = config.pcd_n_sample
                rescale_ratio = config.pcd_rescale_ratio if config.pcd_rescale_ratio is not None else 1.0
                grid_size = config.pcd_grid_size
                self._save_pcd(f"{self.output_dir}/mesh_ply/{iteration}_pcd.ply", rescale_ratio, n_sample, grid_size)

            timer.stop()
            if config.log_interval_iter > 0 and iteration % config.log_interval_iter == 0:
                self.logger.debug(timer.message())

        self.logger.info(timer.message())
        self.logger.info("Training finished")

    def train(self):
        try:
            self._train()
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            del self.dataset
            raise

    def evaluate(self, save_img: bool = False) -> float:
        return self._evaluate(0, use_tensorboard=False, save_img=save_img)

    @torch.no_grad()
    def _save_pcd(self, ply_path: str, rescale_ratio: float = 1.0, n_sample: int = 5_000_000, grid_size: float = None):
        self.logger.info("Rendering point cloud from training views")

        pcd = PointCloud()
        pbar = tqdm(range(self.dataset.getTrainDatasetSize()))
        for i in pbar:
            camera = self.dataset.getTrainData(i).to(self.device)
            camera.image_width = int(camera.image_width * rescale_ratio)
            camera.image_height = int(camera.image_height * rescale_ratio)
            cur_pcd = self.model.render_points(camera)
            pcd += cur_pcd
            pbar.set_postfix_str(f"N points: {len(cur_pcd)}")

        if grid_size is not None:
            self.logger.info(f"Total points: {len(pcd)}, grid sampling with grid size {grid_size}")
            xyz = torch.tensor(pcd.points).to(self.device)
            color = torch.tensor(pcd.colors).to(self.device)
            normal = torch.tensor(pcd.normals).to(self.device)
            pcd = PointCloud(*to_numpy(*grid_sampling(xyz, color, normal, grid_size=grid_size)))

        if n_sample is not None and len(pcd) > n_sample:
            self.logger.info(f"Total points: {len(pcd)}, random sampling to {n_sample} points")
            sample_idx = torch.randperm(len(pcd))[:n_sample]
            pcd = pcd[sample_idx]

        pcd.storePly(ply_path)
