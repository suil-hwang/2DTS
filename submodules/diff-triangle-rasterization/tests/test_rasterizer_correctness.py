"""GPU regressions for the native rasterizer; rebuild the extension before running.

python -m unittest discover -s submodules/diff-triangle-rasterization/tests -v
"""

import importlib
import struct
import unittest

import torch


def reference_pixel(vertices, opacity, features, xy, size, gamma, background, background_depth):
    """Independent differentiable ray/triangle and pairwise-distortion reference.

    Inputs are CPU float64 tensors. Tests use positive depths, strict ordering,
    nondegenerate triangles, and pixels away from barycentric/cutoff ties.
    """
    x, y = xy
    ray = vertices.new_tensor([(2 * x - size + 1) / size, (2 * y - size + 1) / size, 1.0])
    normals = torch.linalg.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    depths = (vertices[:, 0] * normals).sum(-1) / (normals * ray).sum(-1)
    relative = vertices - depths[:, None, None] * ray
    normal_squared = normals.square().sum(-1)
    a1 = (torch.linalg.cross(relative[:, 1], relative[:, 2]) * normals).sum(-1) / normal_squared
    a2 = (torch.linalg.cross(relative[:, 2], relative[:, 0]) * normals).sum(-1) / normal_squared
    barycentric = torch.stack((a1, a2, 1 - a1 - a2), dim=-1)
    eccentricity = 1 - 3 * barycentric.min(dim=-1).values
    profile = torch.exp(-0.5 * eccentricity.pow(2 * gamma))
    alpha = (opacity.flatten() * profile).clamp_max(0.99)
    transmittance = torch.cumprod(torch.cat((alpha.new_ones(1), 1 - alpha[:-1])), dim=0)
    weights = transmittance * alpha
    final_t = (1 - alpha).prod()
    color = (weights[:, None] * features).sum(0) + final_t * vertices.new_tensor(background)
    depth = (weights * depths).sum() + final_t * background_depth
    normal = (weights[:, None] * normals / normal_squared.sqrt()[:, None]).sum(0)
    distortion = weights.new_zeros(())
    for first in range(len(weights)):
        for second in range(first + 1, len(weights)):
            distortion = distortion + weights[first] * weights[second] * (depths[first] - depths[second]).square()
    return color, depth, normal, distortion, final_t, barycentric, depths, profile


class RasterizerCorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("These regression tests require a CUDA GPU and rebuilt extension.")
        cls.dtr = importlib.import_module("diff_triangle_rasterization")

    def tensor(self, values, requires_grad=False):
        return torch.tensor(values, dtype=torch.float32, device="cuda", requires_grad=requires_grad)

    def settings(self, *, size=33, level=0, gamma=1.0, rich=True, background=(0.0, 0.0, 0.0), background_depth=10.0, back_culling=False):
        near, far = 0.01, 100.0
        projection = self.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, far / (far - near), 1.0],
            [0.0, 0.0, -far * near / (far - near), 0.0],
        ])
        return self.dtr.TriangleRasterizationSettings(
            image_width=size, image_height=size, tanfovx=1.0, tanfovy=1.0,
            viewmatrix=torch.eye(4, dtype=torch.float32, device="cuda"),
            projmatrix=projection, campos=self.tensor([0.0, 0.0, 0.0]),
            sh_degree=0, gamma=gamma, background_depth=background_depth,
            background=self.tensor(background), back_culling=back_culling,
            rich_info=rich, sort_level=level, debug=False,
        )

    def render(self, vertices, opacity, features, settings, holder=None):
        if holder is None:
            holder = torch.zeros(vertices.shape[0], device="cuda", requires_grad=True)
        return self.dtr.TriangleRasterizer(settings)(
            vertex=vertices, grad_holder=holder, opacity=opacity, feature=features,
        )

    def centered(self, depths, requires_grad=False):
        return self.tensor([
            [[-0.3 * z, -0.25 * z, z], [0.3 * z, -0.25 * z, z], [0.0, 0.5 * z, z]]
            for z in depths
        ], requires_grad)

    def tilted_inputs(self):
        vertices = self.tensor([
            [[-0.8, -0.6, 1.8], [0.9, -0.5, 2.0], [-0.1, 0.9, 2.2]],
            [[-0.9, -0.75, 2.7], [1.2, -0.5, 2.95], [-0.05, 1.15, 3.15]],
            [[-1.3, -0.95, 3.7], [1.35, -0.8, 4.0], [0.2, 1.5, 4.2]],
        ], True)
        opacity = self.tensor([[0.28], [0.43], [0.36]], True)
        features = self.tensor([[0.8, 0.2, 0.1], [0.15, 0.7, 0.3], [0.25, 0.1, 0.9]], True)
        return vertices, opacity, features

    def assert_close(self, actual, expected, *, atol=3e-5, rtol=3e-4):
        self.assertTrue(torch.isfinite(actual).all().item())
        expected = torch.as_tensor(expected, dtype=actual.dtype, device=actual.device)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def opacity_difference(self, function, opacity, epsilon):
        result = torch.zeros_like(opacity)
        with torch.no_grad():
            for index in range(opacity.numel()):
                original = opacity.flatten()[index].item()
                try:
                    opacity.flatten()[index] = original + epsilon
                    plus = function().item()
                    opacity.flatten()[index] = original - epsilon
                    minus = function().item()
                    result.flatten()[index] = (plus - minus) / (2 * epsilon)
                finally:
                    opacity.flatten()[index] = original
        return result

    def test_distortion_two_quoted_examples_match_finite_difference(self):
        examples = [((0.2, 0.3), (1.0, 3.0)), ((0.3, 0.4), (2.0, 3.0))]
        for level in (0, 1, 2):
            for alphas, depths in examples:
                with self.subTest(level=level, alphas=alphas, depths=depths):
                    vertices = self.centered(depths)
                    opacity = self.tensor([[alphas[0]], [alphas[1]]], True)
                    features = self.tensor([[0.8, 0.3, 0.1], [0.1, 0.3, 0.8]])
                    config = self.settings(level=level)
                    loss = lambda: self.render(vertices, opacity, features, config)[4][16, 16]
                    value = loss()
                    gradient = torch.autograd.grad(value, opacity)[0]
                    numerical = self.opacity_difference(loss, opacity, 1e-3)
                    separation_squared = (depths[0] - depths[1]) ** 2
                    expected_value = alphas[0] * (1 - alphas[0]) * alphas[1] * separation_squared
                    expected_gradient = [
                        [(1 - 2 * alphas[0]) * alphas[1] * separation_squared],
                        [alphas[0] * (1 - alphas[0]) * separation_squared],
                    ]
                    self.assert_close(value, expected_value, atol=2e-6)
                    self.assert_close(gradient, expected_gradient, atol=2e-5)
                    self.assert_close(gradient, numerical, atol=2e-4, rtol=2e-3)

    def test_distortion_zero_and_small_mass_have_finite_correct_gradients(self):
        for level in (0, 1, 2):
            for first, second in ((0.0, 0.0), (0.0, 0.3), (1e-9, 2e-9), (1e-6, 2e-6), (1e-4, 2e-4)):
                with self.subTest(level=level, opacity=(first, second)):
                    vertices = self.centered((2.0, 3.0))
                    opacity = self.tensor([[first], [second]], True)
                    features = self.tensor([[0.8, 0.3, 0.1], [0.1, 0.3, 0.8]])
                    output = self.render(vertices, opacity, features, self.settings(level=level))
                    value = output[4][16, 16]
                    gradient = torch.autograd.grad(value, opacity)[0]
                    expected_value = first * (1 - first) * second
                    expected_gradient = [[(1 - 2 * first) * second], [first * (1 - first)]]
                    gradient_scale = max(abs(component[0]) for component in expected_gradient)
                    # Scale the absolute floor with the expected signal: a zero
                    # result must not pass just because opacity is tiny.
                    self.assert_close(value, expected_value, atol=max(1e-30, abs(expected_value) * 1e-6), rtol=1e-4)
                    self.assert_close(gradient, expected_gradient, atol=max(1e-15, gradient_scale * 1e-6), rtol=1e-4)

    def test_tilted_multilayer_combined_loss_matches_independent_reference(self):
        xy = (18, 14)
        background = (0.07, 0.11, 0.19)
        for level in (0, 1, 2):
            for gamma in (1.0, 1.4):
                with self.subTest(level=level, gamma=gamma):
                    vertices, opacity, features = self.tilted_inputs()
                    reference_inputs = [p.detach().cpu().double().requires_grad_() for p in (vertices, opacity, features)]
                    reference = reference_pixel(*reference_inputs, xy, 33, gamma, background, 6.7)
                    color, depth, normal, distortion, final_t, barycentric, depths, profile = reference
                    ordered_barycentric = barycentric.sort(dim=-1).values
                    self.assertTrue((ordered_barycentric[:, 1] - ordered_barycentric[:, 0] > 0.03).all().item())
                    self.assertTrue((depths.diff() > 0.5).all().item())
                    self.assertTrue((profile > 0.1).all().item())
                    self.assertGreater(final_t.item(), 0.1)
                    color_weight = color.new_tensor([0.7, -0.2, 0.4])
                    normal_weight = color.new_tensor([0.13, -0.07, 0.11])
                    reference_loss = (color * color_weight).sum() + 0.05 * depth + (normal * normal_weight).sum() + 0.35 * distortion
                    reference_gradients = torch.autograd.grad(reference_loss, reference_inputs)
                    output = self.render(vertices, opacity, features, self.settings(level=level, gamma=gamma, background=background, background_depth=6.7))
                    x, y = xy
                    gpu_values = (output[0][:, y, x], output[2][y, x], output[3][:, y, x], output[4][y, x])
                    for actual, expected in zip(gpu_values, reference[:4]):
                        self.assert_close(actual, expected)
                    loss = (gpu_values[0] * color_weight.to("cuda")).sum() + 0.05 * gpu_values[1] + (gpu_values[2] * normal_weight.to("cuda")).sum() + 0.35 * gpu_values[3]
                    gradients = torch.autograd.grad(loss, (vertices, opacity, features))
                    for actual, expected in zip(gradients, reference_gradients):
                        self.assert_close(actual, expected, atol=2e-4, rtol=3e-3)

    def test_alpha_saturation_stops_opacity_gradient_but_keeps_feature_gradient(self):
        for level in (0, 1, 2):
            with self.subTest(level=level):
                vertices = self.centered((2.0,))
                opacity = self.tensor([[0.999]], True)
                features = self.tensor([[0.8, 0.3, 0.1]], True)
                config = self.settings(level=level, background=(0.1, 0.2, 0.3))
                loss = lambda: self.render(vertices, opacity, features, config)[0][0, 16, 16]
                value = loss()
                opacity_gradient, feature_gradient = torch.autograd.grad(value, (opacity, features))
                numerical = self.opacity_difference(loss, opacity, 1e-4)
                self.assert_close(value, 0.793, atol=1e-6)
                self.assert_close(opacity_gradient, [[0.0]], atol=1e-7, rtol=0)
                self.assert_close(opacity_gradient, numerical, atol=1e-7, rtol=0)
                self.assert_close(feature_gradient, [[0.99, 0.0, 0.0]], atol=1e-6)

    def test_binary_ste_composes_with_saturated_and_unsaturated_alpha(self):
        for level in (0, 1, 2):
            for xy in ((16, 16), (18, 16)):
                with self.subTest(level=level, pixel=xy):
                    vertices = self.centered((2.0,))
                    raw_opacity = self.tensor([[0.7]], True)
                    # The model's binary STE is a separate layer: forward one,
                    # derivative one. It does not override the renderer's cap.
                    opacity = ((raw_opacity >= 0.5).float() - raw_opacity).detach() + raw_opacity
                    features = self.tensor([[0.8, 0.3, 0.1]], True)
                    config = self.settings(level=level, background=(0.1, 0.2, 0.3))
                    reference = reference_pixel(
                        vertices.cpu().double(), opacity.detach().cpu().double(), features.detach().cpu().double(),
                        xy, 33, 1.0, (0.1, 0.2, 0.3), 10.0,
                    )
                    profile = reference[7].item()
                    expected_alpha = min(0.99, profile)
                    expected_signal = 0.0 if profile > 0.99 else profile * (0.8 - 0.1)
                    if xy == (16, 16):
                        self.assertGreater(profile, 0.99)
                    else:
                        self.assertLess(profile, 0.95)
                        self.assertGreater(expected_signal, 0.1)
                    x, y = xy
                    value = self.render(vertices, opacity, features, config)[0][0, y, x]
                    opacity_gradient, feature_gradient = torch.autograd.grad(value, (raw_opacity, features))
                    self.assert_close(opacity, [[1.0]], atol=0, rtol=0)
                    self.assert_close(value, reference[0][0], atol=2e-6)
                    # No finite difference through the intentional binary STE:
                    # compare its chain rule with the independently computed G.
                    self.assert_close(opacity_gradient, [[expected_signal]], atol=3e-6, rtol=1e-5)
                    self.assert_close(feature_gradient, [[expected_alpha, 0.0, 0.0]], atol=2e-6)

    def test_degree_zero_sh_at_camera_matches_constant_color_derivatives(self):
        sh_c0 = 0.28209479177387814
        cases = (
            ("face", [[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, -1.0, -2.0]]]),
            ("vertex", [[[0.0, 0.0, 0.0], [0.5, -0.2, 1.0], [0.2, 0.5, 1.0]]]),
        )
        for level in (0, 1, 2):
            for color_mode, geometry in cases:
                with self.subTest(level=level, color_mode=color_mode):
                    config = self.settings(level=level, background=(0.05, 0.1, 0.2))
                    feature_shape = (1, 3, 3) if color_mode == "vertex" else (1, 3)
                    vertices = self.tensor(geometry, True)
                    opacity = self.tensor([[0.3]], True)
                    features = torch.full(feature_shape, 0.5, device="cuda", requires_grad=True)
                    expected_output = self.render(vertices, opacity, features, config)
                    channel_weights = self.tensor([0.7, -0.2, 0.4])[:, None, None]
                    expected_loss = (expected_output[0] * channel_weights).mean()
                    expected_gradients = torch.autograd.grad(expected_loss, (vertices, opacity, features))

                    sh_vertices = self.tensor(geometry, True)
                    sh_opacity = self.tensor([[0.3]], True)
                    sh_shape = (*feature_shape[:-1], 1, 3)
                    shs = torch.zeros(sh_shape, device="cuda", requires_grad=True)
                    holder = torch.zeros(1, device="cuda", requires_grad=True)
                    actual_output = self.dtr.TriangleRasterizer(config)(
                        vertex=sh_vertices, grad_holder=holder, opacity=sh_opacity, shs=shs,
                    )
                    actual_loss = (actual_output[0] * channel_weights).mean()
                    actual_gradients = torch.autograd.grad(actual_loss, (sh_vertices, sh_opacity, shs))

                    # Degree zero is RGB=.5+SH_C0*sh0, independent of direction
                    # even at the camera. Compare the equivalent color paths;
                    # geometry FD would cross unrelated visibility/tie branches.
                    self.assertGreater(expected_output[1][0].item(), 0)
                    self.assertGreater(expected_gradients[1].abs().max().item(), 1e-3)
                    self.assertGreater(expected_gradients[2].abs().max().item(), 1e-3)
                    self.assert_close(actual_output[0], expected_output[0], atol=2e-6, rtol=1e-5)
                    for actual, expected in zip(actual_gradients[:2], expected_gradients[:2]):
                        self.assert_close(actual, expected, atol=2e-6, rtol=3e-4)
                    self.assert_close(
                        actual_gradients[2], sh_c0 * expected_gradients[2].unsqueeze(-2),
                        atol=2e-6, rtol=3e-4,
                    )

    def test_raw_sum_matches_dense_upstream_gradient(self):
        for level in (0, 1, 2):
            with self.subTest(level=level):
                answers = []
                for raw_sum in (False, True):
                    vertices, opacity, features = self.tilted_inputs()
                    output = self.render(vertices, opacity, features, self.settings(level=level))
                    if raw_sum:
                        gradients = torch.autograd.grad(output[0].sum(), (vertices, opacity, features))
                    else:
                        gradients = torch.autograd.grad(output[0], (vertices, opacity, features), grad_outputs=torch.ones_like(output[0]))
                    answers.append(gradients)
                for actual, expected in zip(answers[1], answers[0]):
                    self.assert_close(actual, expected, atol=2e-4, rtol=3e-5)

    def test_opacity_gradient_preserves_small_channel_signal_under_large_cancellation(self):
        for level in (0, 1, 2):
            with self.subTest(level=level):
                vertices = self.centered((2.0,))
                opacity = self.tensor([[0.5]], True)
                features = self.tensor([[1.0, 1.0, 1.0]])
                config = self.settings(size=33, level=level, rich=True, background=(1.0, 0.0, 1.0))
                output = self.render(vertices, opacity, features, config)
                selected = (output[0], output[2], output[3], output[4])
                auxiliary_gradients = tuple(torch.zeros_like(value) for value in selected[1:])
                opacity_gradients = []
                for index, signal in enumerate(((2**24, 1.0, -(2**24)), (0.0, 1.0, 0.0))):
                    color_gradient = torch.zeros_like(output[0])
                    color_gradient[:, 16, 16] = self.tensor(signal)
                    opacity_gradients.append(torch.autograd.grad(
                        selected, opacity, grad_outputs=(color_gradient, *auxiliary_gradients),
                        retain_graph=index == 0,
                    )[0])
                # Red/blue equal their backgrounds, so only green depends on
                # opacity. Prematurely contracting RGB with the large upstream
                # signal can round away green's unit derivative in float32.
                self.assert_close(opacity_gradients[1], [[1.0]], atol=1e-6, rtol=1e-6)
                self.assert_close(opacity_gradients[0], opacity_gradients[1], atol=1e-6, rtol=1e-6)

    def test_long_candidate_prefix_and_partial_tiles_match_rgb_reference(self):
        count, size = 513, 49
        center = size // 2
        background = (0.07, 0.11, 0.19)
        for level in (0, 1):
            for alpha, expected_count in ((0.95, 4), (0.0005, count)):
                with self.subTest(level=level, opacity=alpha):
                    depths = torch.linspace(1.5, 2.5, count, dtype=torch.float32, device="cuda")
                    shape = self.tensor([[-10.0, -10.0, 1.0], [10.0, -10.0, 1.0], [0.0, 20.0, 1.0]])
                    vertices = depths[:, None, None] * shape[None]
                    opacity = torch.full((count, 1), alpha, dtype=torch.float32, device="cuda", requires_grad=True)
                    features = torch.linspace(0.1, 0.9, count * 3, dtype=torch.float32, device="cuda").reshape(count, 3).requires_grad_()
                    output = self.render(vertices, opacity, features, self.settings(size=size, level=level, background=background))
                    # 49x49 includes partial boundary tiles; every triangle
                    # covers every tile, exceeding the 256-candidate batch.
                    self.assertEqual(output[9], count * ((size + 15) // 16) ** 2)
                    self.assertEqual(output[7][center, center].item(), expected_count)
                    self.assertTrue((output[7] == expected_count).all().item())

                    reference_opacity = opacity.detach().cpu().double().requires_grad_()
                    reference_features = features.detach().cpu().double().requires_grad_()
                    active_alpha = reference_opacity[:expected_count, 0]
                    transmittance = torch.cumprod(torch.cat((active_alpha.new_ones(1), 1 - active_alpha[:-1])), dim=0)
                    weights = active_alpha * transmittance
                    final_t = (1 - active_alpha).prod()
                    # At the center, each triangle has G=1. Match forward's
                    # early termination without evaluating pairwise distortion.
                    reference_color = (weights[:, None] * reference_features[:expected_count]).sum(0)
                    reference_color = reference_color + final_t * active_alpha.new_tensor(background)
                    channel_weights = self.tensor([0.7, -0.2, 0.4])
                    reference_loss = (reference_color * channel_weights.cpu().double()).sum()
                    expected_gradients = torch.autograd.grad(reference_loss, (reference_opacity, reference_features))
                    actual_color = output[0][:, center, center]
                    actual_gradients = torch.autograd.grad((actual_color * channel_weights).sum(), (opacity, features))
                    self.assert_close(actual_color, reference_color)
                    for actual, expected in zip(actual_gradients, expected_gradients):
                        self.assert_close(actual, expected)
                        if expected_count < count:
                            # Full-size reference leaves the unvisited tail's
                            # opacity and feature gradients exactly zero.
                            self.assert_close(actual[expected_count:], torch.zeros_like(actual[expected_count:]), atol=0, rtol=0)
                            self.assert_close(expected[expected_count:], torch.zeros_like(expected[expected_count:]), atol=0, rtol=0)

    def test_distortion_upstream_gradient_layout_does_not_change_derivative(self):
        for level in (0, 1, 2):
            with self.subTest(level=level):
                answers = []
                for layout in ("dense", "expanded", "transposed"):
                    vertices = torch.zeros((128, 3, 3), device="cuda")
                    vertices[:2] = self.centered((2.0, 3.0))
                    vertices.requires_grad_()
                    opacity = torch.full((128, 1), 0.3, device="cuda")
                    opacity[1] = 0.4
                    opacity.requires_grad_()
                    features = torch.full((128, 3), 0.5, device="cuda", requires_grad=True)
                    output = self.render(vertices, opacity, features, self.settings(level=level))
                    gradient = self.tensor(1 / (33 * 33)).expand(33, 33)
                    if layout == "dense":
                        gradient = gradient.contiguous()
                    elif layout == "transposed":
                        gradient = gradient.contiguous().transpose(0, 1)
                    self.assertEqual(gradient.is_contiguous(), layout == "dense")
                    selected = (output[0], output[2], output[3], output[4])
                    upstream = (torch.zeros_like(output[0]), torch.zeros_like(output[2]), torch.zeros_like(output[3]), gradient)
                    answers.append(torch.autograd.grad(selected, (vertices, opacity, features), grad_outputs=upstream))
                self.assertGreater(answers[0][0].norm().item(), 1e-4)
                self.assertGreater(answers[0][1][:2].abs().min().item(), 1e-4)
                for candidate in answers[1:]:
                    for actual, expected in zip(candidate, answers[0]):
                        self.assert_close(actual, expected, atol=2e-6, rtol=3e-4)

    def test_empty_scene_uses_background_and_zero_alpha(self):
        background = (0.2, 0.4, 0.6)
        for level in (0, 1, 2):
            for rich in (False, True):
                with self.subTest(level=level, rich=rich):
                    vertices = torch.empty((0, 3, 3), device="cuda")
                    opacity = torch.empty((0, 1), device="cuda")
                    features = torch.empty((0, 3), device="cuda")
                    output = self.render(vertices, opacity, features, self.settings(level=level, rich=rich, background=background, background_depth=7.0))
                    self.assert_close(output[0], self.tensor(background)[:, None, None].expand(3, 33, 33), atol=0, rtol=0)
                    self.assertEqual(output[1].numel(), 0)
                    if rich:
                        self.assert_close(output[2], torch.full_like(output[2], 7.0), atol=0, rtol=0)
                        for index in (3, 4, 7, 8):
                            self.assert_close(output[index], torch.zeros_like(output[index]), atol=0, rtol=0)
                        self.assertEqual(output[9], 0)

    def test_nondefault_stream_forward_and_backward_match_default(self):
        stream = torch.cuda.Stream()
        for level in (0, 1, 2):
            with self.subTest(level=level):
                config = self.settings(level=level, background=(0.1, 0.2, 0.3))
                baseline_inputs = self.tilted_inputs()
                stream_inputs = self.tilted_inputs()
                baseline_holder = torch.zeros(3, device="cuda", requires_grad=True)
                stream_holder = torch.zeros(3, device="cuda", requires_grad=True)
                torch.cuda.synchronize()

                def evaluate(inputs, holder):
                    output = self.render(*inputs, config, holder=holder)
                    loss = output[0].mean() + 0.03 * output[2].mean() + 0.02 * output[3].square().mean() + 0.1 * output[4].mean()
                    gradients = torch.autograd.grad(loss, (*inputs, holder))
                    return output, gradients

                expected_output, expected_gradients = evaluate(baseline_inputs, baseline_holder)
                torch.cuda.synchronize()
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(100_000_000)
                    actual_output, actual_gradients = evaluate(stream_inputs, stream_holder)
                torch.cuda.synchronize()
                for index in (0, 2, 3, 4, 5, 6, 7, 8):
                    self.assert_close(actual_output[index], expected_output[index], atol=2e-5, rtol=3e-4)
                for actual, expected in zip(actual_gradients, expected_gradients):
                    self.assert_close(actual, expected, atol=3e-5, rtol=3e-4)

    def test_centroid_tile_key_uses_centroid_depth(self):
        for level in (0, 1, 2):
            with self.subTest(level=level):
                vertices = self.tensor([[[-0.5, -0.5, 1.5], [0.5, -0.5, 2.5], [0.6, 0.5, 2.6]]], True)
                output = self.render(vertices, self.tensor([[0.5]]), self.tensor([[0.8, 0.3, 0.1]]), self.settings(size=64, level=level))
                buffer = output[0].grad_fn.saved_tensors[-2]
                count = output[9]
                offset = ((buffer.data_ptr() + 127) & ~127) - buffer.data_ptr()
                torch.cuda.synchronize()
                keys = struct.unpack_from(f"<{count}Q", bytes(buffer.detach().cpu().tolist()), offset)
                depths = [struct.unpack("<f", struct.pack("<I", key & 0xffffffff))[0] for key in keys if key >> 32 == 6]
                self.assertEqual(len(depths), 1)
                self.assertAlmostEqual(depths[0], 2.2, delta=2e-6)

    def test_camera_crossing_coverage_and_backface_culling(self):
        vertices = [
            [0.09766795543471284, -0.03003866304138461, 0.5335910869425641],
            [0.12770661847609743, 0.03003866304138461, 0.5335910869425641],
            [0.06462542608918975, 0.0, -0.06718217388512815],
        ]
        for level in (0, 1, 2):
            for reverse in (False, True):
                for back_culling in (False, True):
                    with self.subTest(level=level, reverse=reverse, back_culling=back_culling):
                        ordered = vertices if not reverse else [vertices[0], vertices[2], vertices[1]]
                        geometry = self.tensor([ordered])
                        normal = torch.linalg.cross(geometry[0, 1] - geometry[0, 0], geometry[0, 2] - geometry[0, 0])
                        front_facing = (geometry[0, 0] * normal).sum().item() < 0
                        self.assertEqual(front_facing, not reverse)
                        reference = reference_pixel(geometry.cpu().double(), torch.tensor([[0.8]], dtype=torch.float64), torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64), (55, 31), 64, 1.0, (0.0, 0.0, 0.0), 10.0)
                        self.assertTrue((reference[5] > 0).all().item())
                        self.assertGreater(reference[6].item(), 0.01)
                        expected = reference[0][0] if (not back_culling or front_facing) else 0.0
                        output = self.render(geometry, self.tensor([[0.8]]), self.tensor([[1.0, 0.0, 0.0]]), self.settings(size=64, level=level, back_culling=back_culling))
                        self.assert_close(output[0][0, 31, 55], expected, atol=3e-5)
                        self.assert_close(output[8][31, 55], expected, atol=3e-5)


if __name__ == "__main__":
    unittest.main()
