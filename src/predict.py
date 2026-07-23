"""
Инференс модели на одном пациенте: набор 2D-срезов МРТ -> предсказанный класс деменции.

Пример использования:
    python -m src.predict --slices_dir path/to/patient_slices --weights models/model_alzheimer_3dcnn.pth
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
from skimage.transform import resize

from .config import CONFIG, device, new_h, new_w, target_depth
from .model import load_trained_model
from .preprocessing import normalize_depth, preprocess_3d_tensor

DEFAULT_LABEL_MAP = {
    0: 'Mild Demented',
    1: 'Moderate Demented',
    2: 'Non Demented',
    3: 'Very Mild Demented',
}


def load_volume_from_slices(slices_dir: str) -> np.ndarray:
    """Собирает 3D-объём из отсортированных по имени 2D-срезов в папке."""
    paths = sorted(glob.glob(os.path.join(slices_dir, '*')))
    if not paths:
        raise FileNotFoundError(f'В папке {slices_dir} не найдено срезов')

    slices = []
    for path in paths:
        gray_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        resized_image = resize(gray_image, (new_h, new_w), anti_aliasing=True, mode='reflect')
        slices.append(resized_image)

    volume = normalize_depth(np.stack(slices, axis=0), target_depth)
    return preprocess_3d_tensor(volume)


def predict_one(model, volume: np.ndarray, label_map: dict = DEFAULT_LABEL_MAP):
    tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    return {
        'predicted_class': label_map[pred_idx],
        'probabilities': {label_map[i]: float(p) for i, p in enumerate(probs)},
    }


def main():
    parser = argparse.ArgumentParser(description='Инференс AlzheimerNet3D на одном пациенте')
    parser.add_argument('--slices_dir', required=True, help='Папка со срезами МРТ одного пациента')
    parser.add_argument('--weights', default=CONFIG['model_path'], help='Путь к весам модели (.pth)')
    args = parser.parse_args()

    model = load_trained_model(args.weights, num_classes=len(DEFAULT_LABEL_MAP), device=device)
    volume = load_volume_from_slices(args.slices_dir)
    result = predict_one(model, volume)

    print(f"Предсказанный класс: {result['predicted_class']}")
    print("Вероятности по классам:")
    for cls, p in result['probabilities'].items():
        print(f"  {cls}: {p:.4f}")


if __name__ == '__main__':
    main()
