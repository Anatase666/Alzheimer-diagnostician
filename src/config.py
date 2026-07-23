"""
Общая конфигурация проекта: гиперпараметры, пути, устройство вычислений.
"""
import random

import numpy as np
import torch

CONFIG = {
    # Препроцессинг
    'new_h': 80,
    'new_w': 80,
    'target_depth': 60,

    # Балансировка классов
    'majority_class': 'Non Demented',
    'majority_class_cap': 230,

    # Обучение
    'batch_size': 4,
    'lr_head': 1e-4,
    'lr_finetune': 1e-5,
    'weight_decay': 1e-4,
    'epochs_phase1': 10,
    'epochs_phase2': 10,
    'early_stopping_patience': 5,

    # Пути
    'model_path': 'models/model_alzheimer_3dcnn.pth',
    'pretrained_backbone_path': 'models/r3d_18-b3b3357e.pth',
    'data_pickle_path': 'data.pkl',

    'seed': 42,
}

new_h, new_w = CONFIG['new_h'], CONFIG['new_w']
target_depth = CONFIG['target_depth']
model_path = CONFIG['model_path']


def set_seed(seed: int = CONFIG['seed']) -> None:
    """Фиксирует seed для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


device = get_device()
