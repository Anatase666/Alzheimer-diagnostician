"""
Балансировка классов, разбиение на train/val/test, сохранение тензоров
на диск и DataLoader'ы для обучения/валидации/теста.
"""
import gc

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .augmentation import augment_dataset
from .config import CONFIG, new_h, new_w, target_depth


def make_class_mapping(data: pd.DataFrame) -> dict:
    classes = sorted(data.Class.unique())
    return {cls: i for i, cls in enumerate(classes)}


def prepare_splits(data: pd.DataFrame, dict_class: dict, tensors_dir: str = '.') -> None:
    """
    Оверсэмплинг миноритарных классов, разбиение на train/val/test (70/15/15),
    аугментация train-части и сохранение всех тензоров в *.pth файлы.
    """
    X, y = data.Tensor.tolist(), [dict_class[i] for i in data.Class.tolist()]
    X, y = np.array(X), np.array(y)
    del data
    gc.collect()

    ros = RandomOverSampler(random_state=CONFIG['seed'])
    X_rsmp, y_rsmp = ros.fit_resample(X.reshape(len(X), -1), y)
    X_rsmp = X_rsmp.reshape(-1, target_depth, new_h, new_w)
    del X, y
    gc.collect()

    X_train, X_test_val, y_train, y_test_val = train_test_split(
        X_rsmp, y_rsmp, test_size=0.3, stratify=y_rsmp, random_state=CONFIG['seed'])
    X_test, X_val, y_test, y_val = train_test_split(
        X_test_val, y_test_val, test_size=0.5, stratify=y_test_val, random_state=CONFIG['seed'])
    del X_rsmp, y_rsmp
    gc.collect()

    X_train_tensor = augment_dataset(X_train)                      # (N, D, 1, H, W)
    X_train_tensor = torch.transpose(X_train_tensor, 1, 2).float()  # -> (N, 1, D, H, W)
    torch.save(X_train_tensor, f'{tensors_dir}/X_train_tensor.pth')
    del X_train_tensor
    gc.collect()

    X_val_tensor = torch.tensor(X_val).unsqueeze(1).float()
    torch.save(X_val_tensor, f'{tensors_dir}/X_val_tensor.pth')
    del X_val_tensor
    gc.collect()

    X_test_tensor = torch.tensor(X_test).unsqueeze(1).float()
    torch.save(X_test_tensor, f'{tensors_dir}/X_test_tensor.pth')
    del X_test_tensor
    gc.collect()

    torch.save(torch.tensor(y_train).long(), f'{tensors_dir}/y_train_tensor.pth')
    torch.save(torch.tensor(y_val).long(), f'{tensors_dir}/y_val_tensor.pth')
    torch.save(torch.tensor(y_test).long(), f'{tensors_dir}/y_test_tensor.pth')

    del X_train, X_test, X_test_val, y_train, y_test_val, y_val, y_test
    gc.collect()


def train_dataset_loader(tensors_dir: str = '.') -> DataLoader:
    X_train_tensor = torch.load(f'{tensors_dir}/X_train_tensor.pth')
    y_train_tensor = torch.load(f'{tensors_dir}/y_train_tensor.pth')
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    return DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)


def val_dataset_loader(tensors_dir: str = '.') -> DataLoader:
    X_val_tensor = torch.load(f'{tensors_dir}/X_val_tensor.pth')
    y_val_tensor = torch.load(f'{tensors_dir}/y_val_tensor.pth')
    val_ds = TensorDataset(X_val_tensor, y_val_tensor)
    return DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)


def test_dataset_loader(tensors_dir: str = '.') -> DataLoader:
    X_test_tensor = torch.load(f'{tensors_dir}/X_test_tensor.pth')
    y_test_tensor = torch.load(f'{tensors_dir}/y_test_tensor.pth')
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)
    return DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)
