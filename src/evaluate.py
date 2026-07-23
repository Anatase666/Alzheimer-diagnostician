"""
Оценка обученной модели на тестовой выборке: accuracy, F1, precision,
recall, ROC-AUC (macro, OvR), confusion matrix и classification report.

Запуск:
    python -m src.evaluate
"""
import gc

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import CONFIG, device, model_path
from .dataset import test_dataset_loader
from .model import load_trained_model


def evaluate(model, label_map: dict):
    model.eval()
    y_true, y_pred, y_proba = [], [], []

    test_loader = test_dataset_loader()

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_pred.extend(preds)
            y_proba.extend(probs)
            y_true.extend(y_batch.numpy())

    del test_loader
    gc.collect()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    test_acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
    except ValueError as e:
        roc_auc = float('nan')
        print(f'ROC-AUC не удалось посчитать: {e}')

    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"F1 (macro):     {f1:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro):    {recall:.4f}")
    print(f"ROC-AUC (macro, OvR): {roc_auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    present_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    tick_labels = [label_map[i] for i in present_labels]

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=tick_labels, zero_division=0))

    return {
        'accuracy': test_acc, 'f1_macro': f1, 'precision_macro': precision,
        'recall_macro': recall, 'roc_auc_macro_ovr': roc_auc,
        'confusion_matrix': cm,
    }


def main(label_map: dict, num_classes: int = 4):
    model = load_trained_model(model_path, num_classes=num_classes, device=device)
    return evaluate(model, label_map)


if __name__ == '__main__':
    # Пример: {0: 'Mild Demented', 1: 'Moderate Demented', 2: 'Non Demented', 3: 'Very Mild Demented'}
    default_label_map = {
        0: 'Mild Demented',
        1: 'Moderate Demented',
        2: 'Non Demented',
        3: 'Very Mild Demented',
    }
    main(default_label_map)
