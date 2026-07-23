"""
Обучение AlzheimerNet3D в две фазы:
  1. Фаза 1 — обучается только новый входной слой (stem) и классификатор (fc).
  2. Фаза 2 — fine-tuning всей сети с меньшим learning rate.

Запуск:
    python -m src.train
"""
import copy
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from .config import CONFIG, device, model_path, set_seed
from .dataset import train_dataset_loader, val_dataset_loader
from .model import AlzheimerNet3D, set_phase1_trainable, set_phase2_trainable


def run_training_phase(model, phase_name, num_epochs, lr, history, best_state):
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=CONFIG['weight_decay'],
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(2, num_epochs // 3))
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_loader = train_dataset_loader()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100 * correct / total
        del train_loader
        gc.collect()

        model.eval()
        val_loader = val_dataset_loader()
        running_val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = running_val_loss / total
        val_acc = 100 * correct / total
        del val_loader
        gc.collect()

        scheduler.step(epoch)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        improved = val_loss < best_state['loss']
        if improved:
            best_state['loss'] = val_loss
            best_state['weights'] = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), model_path)
            epochs_without_improvement = 0
            marker = ' (новая лучшая модель сохранена)'
        else:
            epochs_without_improvement += 1
            marker = ''

        print(f"[{phase_name}] Эпоха {epoch + 1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}{marker}")

        if epochs_without_improvement >= CONFIG['early_stopping_patience']:
            print(f"[{phase_name}] Early stopping: нет улучшений "
                  f"{CONFIG['early_stopping_patience']} эпох подряд.")
            break

    if best_state['weights'] is not None:
        model.load_state_dict(best_state['weights'])
    return model


def main(num_classes: int = 4):
    set_seed()

    model = AlzheimerNet3D(
        num_classes=num_classes,
        weights_path=CONFIG['pretrained_backbone_path'],
    ).to(device)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_state = {'loss': float('inf'), 'weights': None}

    set_phase1_trainable(model)
    model = run_training_phase(
        model, phase_name='Фаза 1 (голова)',
        num_epochs=CONFIG['epochs_phase1'], lr=CONFIG['lr_head'],
        history=history, best_state=best_state,
    )

    set_phase2_trainable(model)
    best_state['loss'] = float('inf')
    model = run_training_phase(
        model, phase_name='Фаза 2 (fine-tuning)',
        num_epochs=CONFIG['epochs_phase2'], lr=CONFIG['lr_finetune'],
        history=history, best_state=best_state,
    )

    torch.save(model.state_dict(), model_path)
    print(f'Финальные веса сохранены в {model_path}')
    return model, history


if __name__ == '__main__':
    main()
