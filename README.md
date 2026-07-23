# Интеллектуальная медицинская система диагностики болезни Альцгеймера (ИМСДА)

3D-CNN на базе `r3d_18` (3D ResNet-18), классифицирующая МРТ-снимки головного
мозга по четырём стадиям деменции:

- Нет деменции (Non Demented)
- Очень легкая деменция (Very Mild Demented)
- Легкая деменция (Mild Demented)
- Умеренная деменция (Moderate Demented)

Проект реализует техническое задание [`docs/Technical_assignment.pdf`](docs/Technical_assignment.pdf)
(автор: Горинов Ярослав, НИТУ МИСИС).

## Структура репозитория

```
.
├── docs/
│   └── Technical_assignment.pdf      # Техническое задание
├── models/
│   └── model_alzheimer_3dcnn.pth     # Обученные веса модели (через Git LFS)
├── notebooks/
│   └── training_notebook.ipynb       # Исходный ноутбук с полным пайплайном обучения
├── src/
│   ├── config.py                     # Гиперпараметры, пути, device
│   ├── preprocessing.py              # Сборка 3D-объёмов из срезов МРТ, нормализация
│   ├── augmentation.py               # Аугментации 3D-объёмов
│   ├── dataset.py                    # Балансировка классов, train/val/test split, DataLoader'ы
│   ├── model.py                      # Архитектура AlzheimerNet3D (r3d_18)
│   ├── train.py                      # Обучение (Фаза 1: голова, Фаза 2: fine-tuning)
│   ├── evaluate.py                   # Оценка на тестовой выборке (accuracy, F1, ROC-AUC, confusion matrix)
│   └── predict.py                    # Инференс на новом пациенте
├── requirements.txt
├── .gitattributes                    # Git LFS для весов модели
├── .gitignore
├── LICENSE
└── README.md
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Для работы с весами модели через Git LFS (файл `models/model_alzheimer_3dcnn.pth`
весит ~127 МБ, что превышает лимит GitHub в 100 МБ для обычных файлов):

```bash
git lfs install
```

## Данные

Обучение использует датасет [OASIS](https://www.kaggle.com/datasets/ninadaithal/imagesoasis)
(`ninadaithal/imagesoasis`, Kaggle), загружаемый через `kagglehub`.
В качестве предобученного backbone используются веса `r3d_18` (Kinetics-400).

## Обучение и оценка

```bash
python -m src.train        # обучение модели (Фаза 1 + Фаза 2), сохраняет models/model_alzheimer_3dcnn.pth
python -m src.evaluate      # метрики на тестовой выборке
python -m src.predict --slices_dir <папка со срезами пациента>   # инференс на новом пациенте
```

Полный пайплайн (загрузка данных, препроцессинг, EDA, обучение, оценка) также
доступен в `notebooks/training_notebook.ipynb`.

## Основные характеристики (по ТЗ)

- **Вход:** серии МРТ-снимков в формате DICOM/NIFTI/JPEG, 61×61 пикс. (в пайплайне ресайз до 80×80, 60 срезов на объём)
- **Выход:** вероятностная оценка принадлежности к одной из 4 категорий деменции
- **Время анализа:** не более 1 минуты на пациента
- **Язык реализации:** Python 3.8+
- **Стек:** PyTorch / torchvision (3D CNN), OpenCV, scikit-image

Подробные требования к функциональности, надёжности, условиям эксплуатации,
аппаратным средствам и программной совместимости — см. [техническое задание](docs/Technical_assignment.pdf).

## Лицензия

[MIT](LICENSE)
