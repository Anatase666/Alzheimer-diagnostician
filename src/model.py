"""
Архитектура модели: r3d_18 (3D ResNet-18 из torchvision), адаптированная
под одноканальный вход (MRI) и заданное число классов деменции.
"""
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18


class AlzheimerNet3D(nn.Module):
    """
    3D-CNN на основе backbone r3d_18.

    Первый свёрточный слой заменён на однокaнальный вход (вместо RGB),
    финальный полносвязный слой — на классификатор с num_classes выходами.
    """

    def __init__(self, num_classes: int = 4, weights_path: str | None = None):
        super().__init__()

        self.backbone = r3d_18(weights=None)

        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict)
            print(f"Веса загружены из: {weights_path}")
        else:
            print("Веса не загружены. Модель инициализирована случайно.")

        self.backbone.stem[0] = nn.Conv3d(
            1, 64, kernel_size=(7, 7, 7),
            stride=(3, 3, 3), padding=(2, 2, 2), bias=False,
        )

        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features, num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def set_phase1_trainable(model: AlzheimerNet3D) -> None:
    """Фаза 1: обучаем только новый stem и голову классификатора."""
    for name, param in model.named_parameters():
        param.requires_grad = ('stem.0' in name) or ('fc' in name)


def set_phase2_trainable(model: AlzheimerNet3D) -> None:
    """Фаза 2: fine-tuning всей сети целиком."""
    for param in model.parameters():
        param.requires_grad = True


def load_trained_model(weights_path: str, num_classes: int = 4, device: str | torch.device = 'cpu') -> AlzheimerNet3D:
    """Загружает архитектуру и обученные веса (например, model_alzheimer_3dcnn.pth) для инференса."""
    model = AlzheimerNet3D(num_classes=num_classes, weights_path=None)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
