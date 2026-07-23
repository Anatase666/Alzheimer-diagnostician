"""
Препроцессинг данных: сборка 2D-срезов MRI в 3D-объёмы фиксированного размера
и их нормализация.

Ожидаемая структура исходных данных (например, датасет OASIS с Kaggle,
``ninadaithal/imagesoasis``):

    dataset_path/
        <Class name>/
            <patient>_<...>_slice.jpg
            ...

Срезы одного пациента группируются по общему префиксу имени файла.
"""
import gc
import os

import cv2
import numpy as np
import pandas as pd
from skimage.transform import resize

from .config import CONFIG, new_h, new_w, target_depth


def build_file_index(dataset_path: str) -> pd.DataFrame:
    """Строит таблицу path/name/class по всем файлам в dataset_path."""
    data = []
    for stage_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, stage_name)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            data.append({
                "Path": img_path,
                'Name': img_file,
                "Class": stage_name,
            })

    df = pd.DataFrame(data)
    df = df.sort_values('Name')
    return df


def normalize_depth(volume: np.ndarray, depth: int) -> np.ndarray:
    """Приводит 3D-объём (D, H, W) к фиксированному числу срезов depth."""
    d = volume.shape[0]
    if d == depth:
        return volume
    if d > depth:
        start = (d - depth) // 2
        return volume[start:start + depth]
    pad_before = (depth - d) // 2
    pad_after = depth - d - pad_before
    return np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)), mode='edge')


def build_volumes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Группирует срезы одного пациента (по первым 4 частям имени файла)
    в один 3D-объём, приводит его к target_depth срезам и ресайзит
    каждый срез до (new_h, new_w).

    Ограничивает число объектов мажоритарного класса (CONFIG['majority_class'])
    величиной CONFIG['majority_class_cap'] для балансировки датасета.
    """
    majority_class = CONFIG['majority_class']
    majority_cap = CONFIG['majority_class_cap']

    results = []
    current_code, current_class, current_slices = None, None, []
    class_counts = {}

    for row in df.to_dict('records'):
        code_ = '_'.join(row['Name'].split('_')[:4])

        if code_ != current_code:
            if current_slices:
                under_cap = not (current_class == majority_class
                                  and class_counts.get(current_class, 0) >= majority_cap)
                if under_cap:
                    volume = normalize_depth(np.stack(current_slices, axis=0), target_depth)
                    results.append({"Tensor": volume, "Class": current_class})
                    class_counts[current_class] = class_counts.get(current_class, 0) + 1
            current_code = code_
            current_class = row['Class']
            current_slices = []

        gray_image = cv2.imread(row['Path'], cv2.IMREAD_GRAYSCALE)
        resized_image = resize(gray_image, (new_h, new_w), anti_aliasing=True, mode='reflect')
        current_slices.append(resized_image)
        current_class = row['Class']

    if current_slices:
        under_cap = not (current_class == majority_class
                          and class_counts.get(current_class, 0) >= majority_cap)
        if under_cap:
            volume = normalize_depth(np.stack(current_slices, axis=0), target_depth)
            results.append({"Tensor": volume, "Class": current_class})

    data = pd.DataFrame(results)
    del results, current_code, current_class, current_slices, class_counts
    gc.collect()
    return data


def preprocess_3d_tensor(tensor: np.ndarray) -> np.ndarray:
    """Клиппинг интенсивностей и нормализация объёма в диапазон [0, 1]."""
    tensor = np.clip(tensor, 0, 250)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return tensor


def build_dataset(dataset_path: str, save_pickle_path: str | None = None) -> pd.DataFrame:
    """Полный пайплайн: индекс файлов -> 3D-объёмы -> нормализация -> (опц.) pickle."""
    df = build_file_index(dataset_path)
    data = build_volumes(df)
    data['Tensor'] = data['Tensor'].apply(preprocess_3d_tensor)

    if save_pickle_path:
        data.to_pickle(save_pickle_path)

    return data
