import torch
import random
import numpy as np


class NormalizeTransform:
    def __call__(self, x):
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)


class AddNoiseTransform:
    def __init__(self, noise_level=0.01):
        self.noise_level = noise_level

    def __call__(self, x):
        noise = torch.randn_like(x) * self.noise_level
        return x + noise


class RandomCropTransform:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, x):
        if x.size(-1) <= self.crop_size:
            return x
        start = random.randint(0, x.size(-1) - self.crop_size)
        return x[..., start:start + self.crop_size]


class RandomTimeWarpTransform:
    def __call__(self, x):
        time_steps = np.arange(x.shape[-1])
        random_factor = np.random.uniform(0.8, 1.2, size=x.shape[-1])
        warped_time_steps = (time_steps * random_factor).astype(int)
        warped_time_steps = np.clip(warped_time_steps, 0, x.shape[-1] - 1)
        return x[..., warped_time_steps]


class RandomScalingTransform:
    def __call__(self, x):
        scaling_factor = np.random.uniform(0.8, 1.2)
        return x * scaling_factor


class RandomErasingTransform:
    def __init__(self, p=0.5, scale=(0.02, 0.33)):
        self.p = p
        self.scale = scale

    def __call__(self, x):
        if random.uniform(0, 1) > self.p:
            return x
        num_channels, seq_len = x.size()
        erase_len = int(random.uniform(*self.scale) * seq_len)
        start = random.randint(0, seq_len - erase_len)
        x[:, start:start + erase_len] = 0
        return x
