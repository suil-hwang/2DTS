import torch
import torch.nn.functional as F
import numpy as np

from simple_knn import distCUDA2


def inverse_sigmoid(x: torch.Tensor | float) -> torch.Tensor:
    x = torch.as_tensor(x)
    if not x.is_floating_point():
        x = x.float()
    if not torch.isfinite(x).all() or ((x < 0) | (x > 1)).any():
        raise ValueError("inverse_sigmoid input must be finite and between 0 and 1")
    zero, one = x.new_tensor(0.0), x.new_tensor(1.0)
    # A normal lower bound also stays finite on CUDA kernels that flush subnormals.
    x = x.clamp(torch.finfo(x.dtype).tiny, torch.nextafter(one, zero))
    return torch.log(x) - torch.log1p(-x)


def to_numpy(*args: torch.Tensor) -> tuple[np.ndarray, ...]:
    return tuple(arg.detach().cpu().numpy() for arg in args)


def cat_tensor_dict(d1: dict[str, torch.Tensor], d2: dict[str, torch.Tensor], dim=0):
    for k, v in d2.items():
        d1[k] = torch.cat((d1[k], v), dim=dim) if k in d1 else v


def filter_tensor_dict(d: dict[str, torch.Tensor], indices: torch.Tensor):
    for k, v in d.items():
        d[k] = v[indices]


def get_first_unique_indices(t: torch.Tensor, dim=0) -> torch.Tensor:
    _, idx, counts = torch.unique(t, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0) - counts
    first_indicies = ind_sorted[cum_sum]
    return first_indicies


def inter_point_distance(pc: torch.Tensor) -> torch.Tensor:
    assert pc.dim() == 2 and pc.size(1) == 3
    if not pc.is_floating_point() or not torch.isfinite(pc).all():
        raise ValueError("point coordinates must be finite floating point values")
    point_count = pc.shape[0]
    if point_count == 0:
        return pc.new_empty(0)
    if point_count == 1:
        return pc.new_full((1,), 1e-5)
    if point_count <= 3:
        # The native kernel always averages three neighbors and otherwise leaves
        # FLT_MAX entries for tiny clouds. Average the available neighbors here.
        distances_squared = (pc[:, None, :] - pc[None, :, :]).square().sum(dim=-1)
        distances_squared.fill_diagonal_(0)
        return (distances_squared.sum(dim=1) / (point_count - 1)).clamp(min=1e-10).sqrt()
    return distCUDA2(pc).clamp_(min=1e-10).sqrt()


def get_inside_mask(points: torch.Tensor, bbox: tuple[float, ...]) -> torch.Tensor:
    if bbox is not None:
        if len(bbox) == 4:
            x_min, y_min, x_max, y_max = bbox
            mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        elif len(bbox) == 6:
            x_min, y_min, z_min, x_max, y_max, z_max = bbox
            mask = (
                (points[:, 0] >= x_min)
                & (points[:, 0] <= x_max)
                & (points[:, 1] >= y_min)
                & (points[:, 1] <= y_max)
                & (points[:, 2] >= z_min)
                & (points[:, 2] <= z_max)
            )
        else:
            raise ValueError(f"bbox must be of length 4 or 6, but got {len(bbox)}")
    else:
        mask = torch.ones_like(points[:, 0], dtype=torch.bool)
    return mask


def argsort(keys: torch.Tensor, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
    indices = torch.argsort(keys)
    return tuple(arg[indices] for arg in args)


def get_color_tensor(color: str) -> torch.Tensor:
    if color == "black":
        return torch.tensor([0, 0, 0], dtype=torch.float32)
    elif color == "white":
        return torch.tensor([1, 1, 1], dtype=torch.float32)
    elif color == "random":
        return torch.rand(3)
    else:
        raise ValueError(f"Unknown background color: {color}")


def grid_sampling(xyz: torch.Tensor, *attrs: torch.Tensor, grid_size: float = 0.0) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """
    :param xyz: tensor with shape (N, 3), where N is the number of points
    :param attrs: tuple of tensors with shape (N, D)
    :param grid_size: float
    :return: (M, 3) or ((M, 3), ...), where M is the number of sampled points
    """
    if not np.isfinite(grid_size) or grid_size < 0:
        raise ValueError("grid_size must be finite and nonnegative")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not xyz.is_floating_point() or not torch.isfinite(xyz).all():
        raise ValueError("xyz must contain finite floating point coordinates with shape (N, 3)")
    for attr in attrs:
        if attr.ndim != 2 or attr.shape[0] != xyz.shape[0] or attr.device != xyz.device:
            raise ValueError("grid attributes must have shape (N, D) on the same device as xyz")
    if grid_size == 0.0:
        return xyz if len(attrs) == 0 else (xyz, *attrs)

    rounded = torch.round(xyz / grid_size)
    if not torch.isfinite(rounded).all() or (rounded.abs() >= 2 ** 63).any():
        raise ValueError("grid_size is too small for these coordinates")
    grid_coords = rounded.long()
    if len(attrs) == 0:
        grid_coords_unique = torch.unique(grid_coords, dim=0)
        sampled_xyz = grid_coords_unique.to(xyz.dtype) * grid_size
        return sampled_xyz
    else:
        grid_coords_unique, inverse_indices = torch.unique(grid_coords, return_inverse=True, dim=0)
        sampled_xyz = grid_coords_unique.to(xyz.dtype) * grid_size
        sampled_attrs = []
        for attr in attrs:
            dtype = attr.dtype if attr.is_floating_point() else torch.float32
            sampled_attr = torch.zeros((sampled_xyz.shape[0], attr.shape[1]), dtype=dtype, device=attr.device)
            sampled_attr.scatter_reduce_(0, inverse_indices.unsqueeze(1).expand(-1, attr.shape[1]), attr.to(dtype), "mean", include_self=False)
            sampled_attrs.append(sampled_attr)
        return sampled_xyz, *sampled_attrs


def grid_size_search(xyz: torch.Tensor, n_sample: int, tolerance: float = 0.1, max_retry: int = 10) -> float:
    """
    Find a grid size whose sampled count is near n_sample, on a best-effort basis.
    Grid counts are not monotone in grid size, so the target tolerance is not
    guaranteed. Keep the closest evaluated count, including the unchanged input
    at grid size zero. With no retries, return zero.

    :param xyz: (N, 3)
    :param n_sample: positive target count, or None to keep all points
    :return: float
    """
    if n_sample is not None and (isinstance(n_sample, bool) or not isinstance(n_sample, int) or n_sample <= 0):
        raise ValueError("n_sample must be a positive integer or None")
    if not np.isfinite(tolerance) or tolerance < 0 or isinstance(max_retry, bool) or not isinstance(max_retry, int) or max_retry < 0:
        raise ValueError("tolerance and max_retry must be nonnegative")
    if n_sample is None or n_sample >= xyz.shape[0] or max_retry == 0:
        return 0.0

    min_grid_size = 0
    max_grid_size = (xyz.max(dim=0).values - xyz.min(dim=0).values).max().item()
    if max_grid_size == 0:
        if abs(1 - n_sample) < xyz.shape[0] - n_sample:
            return max(xyz.abs().max().item(), 1.0) * torch.finfo(xyz.dtype).eps
        return 0.0
    n_sample_error_tolerance = tolerance * n_sample
    n_sample_min = n_sample - n_sample_error_tolerance
    n_sample_max = n_sample + n_sample_error_tolerance

    grid_size = max_grid_size / n_sample ** (1 / 3)
    best_grid_size = 0.0
    best_error = xyz.shape[0] - n_sample

    for _ in range(max_retry):
        n = grid_sampling(xyz, grid_size=grid_size).shape[0]
        error = abs(n - n_sample)
        if error < best_error:
            best_grid_size = grid_size
            best_error = error
        if n_sample_min <= n <= n_sample_max:
            return best_grid_size
        elif n < n_sample_min:
            max_grid_size = grid_size
            grid_size = (min_grid_size + max_grid_size) / 2
        else:
            min_grid_size = grid_size
            grid_size = (min_grid_size + max_grid_size) / 2
    return best_grid_size


def binary_erosion(image_tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Performs binary erosion on a PyTorch tensor using convolution.

    Args:
        image_tensor (torch.Tensor): Input binary image tensor (0s and 1s).
                                     Expected shape: (N, C, H, W), (C, H, W) or (H, W).
        kernel_size (int or tuple): Size of the square structuring element.

    Returns:
        torch.Tensor: Eroded binary image tensor.
    """
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size % 2 != 1:
        raise ValueError("kernel_size must be a positive odd integer")
    if image_tensor.ndim not in (2, 3, 4):
        raise ValueError("image_tensor must have 2, 3, or 4 dimensions")

    input_shape = image_tensor.shape
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    elif image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Create a kernel of ones for the structuring element
    channels = image_tensor.shape[1]
    kernel = torch.ones(channels, 1, kernel_size, kernel_size, device=image_tensor.device)

    # Invert the image (foreground becomes 0, background becomes 1)
    inverted_image = 1 - image_tensor.float()

    # Perform convolution on the inverted image
    # Padding is crucial to maintain image size
    padding = kernel_size // 2
    convolved_inverted = F.conv2d(inverted_image, kernel, padding=padding, groups=channels)

    # Threshold the convolved result and invert back to get eroded image
    # A pixel is eroded if any part of the structuring element overlaps with background (0 in original)
    # So, if convolved_inverted > 0, it means there was a background pixel in the neighborhood,
    # and the corresponding pixel in the eroded image should be 0.
    eroded_image = (convolved_inverted == 0)
    eroded_image = eroded_image.view(input_shape)  # Restore original shape
    return eroded_image


if __name__ == "__main__":
    xyz = torch.rand(10_000, 3) * 1000
    shs = torch.rand(10_000, 10)
    grid_size = 1
    sampled_xyz, sampled_shs = grid_sampling(xyz, shs, grid_size=grid_size)
    # grid_size_search(xyz, 1_000_000)
