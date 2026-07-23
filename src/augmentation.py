"""
Аугментация 3D MRI-объёмов: случайный горизонтальный флип, поворот
и добавление гауссовского шума, применяемые согласованно ко всем срезам
одного объёма.
"""
import random

import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class AddGaussianNoise(nn.Module):
    def __init__(self, mean: float = 0., std: float = 1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(img) * self.std + self.mean
        return img + noise


def augment_volume(volume, noise_std: float = 0.005) -> torch.Tensor:
    """Применяет одинаковый флип/поворот ко всем срезам объёма + шум на каждый срез."""
    do_flip = random.random() < 0.5
    angle = random.uniform(-90, 90)
    noiser = AddGaussianNoise(std=noise_std)

    augmented_slices = []
    for frame in volume:
        tensor = torch.from_numpy(frame).float().unsqueeze(0)  # (1, H, W)
        if do_flip:
            tensor = F.hflip(tensor)
        tensor = F.rotate(tensor, angle)
        tensor = noiser(tensor)
        augmented_slices.append(tensor)

    return torch.stack(augmented_slices, dim=0)  # (D, 1, H, W)


def augment_dataset(volumes) -> torch.Tensor:
    return torch.stack([augment_volume(v) for v in volumes], dim=0)  # (N, D, 1, H, W)
