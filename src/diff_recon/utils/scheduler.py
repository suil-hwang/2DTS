import numpy as np
from typing import List


def exponential_scheduler(v_init: float, v_final: float, max_steps: int, delay_steps: int = 0, delay_mult: float = 1.0):
    if not np.isfinite(v_init) or not np.isfinite(v_final) or v_init < 0 or v_final < 0:
        raise ValueError("Exponential schedule endpoints must be finite and nonnegative")
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")

    def scheduler(step: int):
        if step <= 0:
            return v_init
        if step >= max_steps:
            return v_final

        if delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = delay_mult + (1 - delay_mult) * np.sin(0.5 * np.pi * np.clip(step / delay_steps, 0, 1))
        else:
            delay_rate = 1.0

        t = np.clip(step / max_steps, 0, 1)
        # The existing geometric interpolation is zero between endpoints when
        # either endpoint is zero; handle it without taking log(0).
        log_lerp = 0.0 if v_init == 0 or v_final == 0 else np.exp(np.log(v_init) * (1 - t) + np.log(v_final) * t)
        return delay_rate * log_lerp

    return scheduler


def linear_scheduler(v_init: float, v_final: float, max_steps: int):

    def scheduler(step: int):
        if step <= 0:
            return v_init
        if step >= max_steps:
            return v_final

        t = np.clip(step / max_steps, 0, 1)
        lerp = v_init * (1 - t) + v_final * t
        return lerp

    return scheduler


def step_scheduler(v_list: List[float], step_list: List[int]):
    if not v_list or len(v_list) not in (len(step_list), len(step_list) + 1):
        raise ValueError("A step schedule needs nonempty values matching its boundaries")
    if any(first > second for first, second in zip(step_list, step_list[1:])):
        raise ValueError("Step boundaries must be in nondecreasing order")

    def scheduler(step: int):
        for i, s in enumerate(step_list):
            if step < s:
                return v_list[i]
        return v_list[-1]

    return scheduler


def exponential_step_scheduler(v_init: float, v_final: float, max_steps: int, n_stage: int, delay_steps: int = 0, delay_mult: float = 1.0):
    if not isinstance(n_stage, (int, np.integer)) or n_stage <= 0:
        raise ValueError("n_stage must be a positive integer")
    exp_scheduler_ = exponential_scheduler(v_init, v_final, max_steps, delay_steps, delay_mult)
    if max_steps == 0:
        return exp_scheduler_
    step_list = [int(max_steps * i / n_stage) for i in range(1, n_stage + 1)]
    v_list = [v_init] + [exp_scheduler_(step) for step in step_list]
    return step_scheduler(v_list, step_list)
