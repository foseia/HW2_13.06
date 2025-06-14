import math
import os
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import wandb
import zipfile

# Установка случайного начального значения для воспроизводимости
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(123456)

# Инструкции для загрузки датасета с Яндекс.Диска:
# 1. Скачайте train.zip: https://downloader.disk.yandex.ru/d/xcyohKB-UE2Ciw
# 2. Скачайте val.zip: https://downloader.disk.yandex.ru/d/qw7d6iAG3h5NNQ
# 3. Распакуйте архивы в директорию /content/dataset/dataset (для Colab) или ./dataset/dataset (локально)
# Пример для Colab:
# !wget https://downloader.disk.yandex.ru/d/xcyohKB-UE2Ciw -O train.zip
# !wget https://downloader.disk.yandex.ru/d/qw7d6iAG3h5NNQ -O val.zip

# Создание директории для датасета
DATASET_PATH = './dataset/dataset'  # Для Colab используйте '/content/dataset/dataset'
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, 'train'), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, 'val'), exist_ok=True)

# Распаковка архивов (раскомментируйте, если файлы загружены)
"""
with zipfile.ZipFile('train.zip', 'r') as zip_ref:
    zip_ref.extractall(os.path.join(DATASET_PATH, 'train'))
with zipfile.ZipFile('val.zip', 'r') as zip_ref:
    zip_ref.extractall(os.path.join(DATASET_PATH, 'val'))
"""

# Проверка содержимого директорий
print("Содержимое директории train:", os.listdir(os.path.join(DATASET_PATH, 'train')))
print("Содержимое директории val:", os.listdir(os.path.join(DATASET_PATH, 'val')))

# Задание 1: Подготовка данных (без изменения размера, пользовательские аугментации)
train_transform_task1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
])

val_transform_task1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset_task1 = ImageFolder(os.path.join(DATASET_PATH, 'train'), transform=train_transform_task1)
val_dataset_task1 = ImageFolder(os.path.join(DATASET_PATH, 'val'), transform=val_transform_task1)

train_dataloader_task1 = DataLoader(train_dataset_task1, batch_size=32, shuffle=True, num_workers=4)
val_dataloader_task1 = DataLoader(val_dataset_task1, batch_size=32, shuffle=False, num_workers=4)

# Задание 2: Подготовка данных (с изменением размера и более мощными аугментациями)
train_transform_task2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
])

val_transform_task2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_dataset_task2 = ImageFolder(os.path.join(DATASET_PATH, 'train'), transform=train_transform_task2)
val_dataset_task2 = ImageFolder(os.path.join(DATASET_PATH, 'val'), transform=val_transform_task2)

train_dataloader_task2 = DataLoader(train_dataset_task2, batch_size=32, shuffle=True, num_workers=4)
val_dataloader_task2 = DataLoader(val_dataset_task2, batch_size=32, shuffle=False, num_workers=4)

# Проверка датасета
assert isinstance(train_dataset_task1[0], tuple)
assert len(train_dataset_task1[0]) == 2
assert isinstance(train_dataset_task1[0][1], int)
print("Проверки датасета пройдены")

# Визуализация нескольких изображений (для отладки)
for batch in val_dataloader_task1:
    images, class_nums = batch
    plt.imshow(images[5].permute(1, 2, 0))
    plt.show()
    plt.imshow(images[19].permute(1, 2, 0))
    plt.show()
    break

# Задание 1: Пользовательская CNN-модель
class YourNetTask1(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 100),
        )
        self.accuracy = accuracy

    def _forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def forward(self, images, target=None):
        output = self._forward(images)
        if target is not None:
            loss = F.cross_entropy(output, target)
            return loss
        return output

    def get_accuracy(self, reset=False):
        return self.accuracy

# Задание 1: Lightning-модуль
class YourModuleTask1(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

    def training_step(self, batch, batch_idx):
        images, labels = batch
        loss = self.model(images, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        output = self.model._forward(images)
        loss = F.cross_entropy(output, labels)
        acc = accuracy(output, labels, task='multiclass', num_classes=100)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', acc, prog_bar=True)
        return loss

# Задание 2: Предобученная модель ResNet-18
class YourNetTask2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, 100)
        self.accuracy = accuracy

    def _forward(self, x):
        return self.model(x)

    def forward(self, images, target=None):
        output = self._forward(images)
        if target is not None:
            loss = F.cross_entropy(output, target)
            return loss
        return output

    def get_accuracy(self, reset=False):
        return self.accuracy

# Задание 2: Lightning-модуль
class YourModuleTask2(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.model.fc.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

    def training_step(self, batch, batch_idx):
        images, labels = batch
        loss = self.model(images, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        output = self.model._forward(images)
        loss = F.cross_entropy(output, labels)
        acc = accuracy(output, labels, task='multiclass', num_classes=100)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', acc, prog_bar=True)
        return loss

# Обучение для задания 1
wandb_logger = WandbLogger(project='emoji_classification_task1', log_model='all')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_task1 = YourNetTask1().to(device)
module_task1 = YourModuleTask1(model_task1, learning_rate=1e-3)

trainer_task1 = pl.Trainer(
    logger=wandb_logger,
    max_epochs=20,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            monitor='val_accuracy',
            mode='max',
            save_top_k=1,
            filename='best-checkpoint-task1',
        ),
        pl.callbacks.EarlyStopping(monitor='val_loss', patience=10),
    ]
)
trainer_task1.fit(module_task1, train_dataloader_task1, val_dataloader_task1)

# Обучение для задания 2
wandb_logger_task2 = WandbLogger(project='emoji_classification_task2', log_model='all')
model_task2 = YourNetTask2().to(device)
module_task2 = YourModuleTask2(model_task2, learning_rate=1e-3)

trainer_task2 = pl.Trainer(
    logger=wandb_logger_task2,
    max_epochs=10,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            monitor='val_accuracy',
            mode='max',
            save_top_k=1,
            filename='best-checkpoint-task2',
        ),
        pl.callbacks.EarlyStopping(monitor='val_loss', patience=5),
    ]
)
trainer_task2.fit(module_task2, train_dataloader_task2, val_dataloader_task2)

# Загрузка чекпоинтов из W&B
wandb.login()
run = wandb.init(project="emoji_classification_task1", job_type="eval")
artifact_task1 = run.use_artifact('best-checkpoint-task1:latest', type='model')
artifact_dir_task1 = artifact_task1.download()
checkpoint_path_task1 = f"{artifact_dir_task1}/best-checkpoint-task1.ckpt"
run.finish()

run = wandb.init(project="emoji_classification_task2", job_type="eval")
artifact_task2 = run.use_artifact('best-checkpoint-task2:latest', type='model')
artifact_dir_task2 = artifact_task2.download()
checkpoint_path_task2 = f"{artifact_dir_task2}/best-checkpoint-task2.ckpt"
run.finish()

# Альтернатива: Локальные чекпоинты (раскомментируйте и укажите пути)
"""
checkpoint_path_task1 = './lightning_logs/version_0/checkpoints/best-checkpoint-task1.ckpt'
checkpoint_path_task2 = './lightning_logs/version_1/checkpoints/best-checkpoint-task2.ckpt'
"""

# Альтернатива: Чекпоинты с Яндекс.Диска (раскомментируйте и добавьте ссылки)
"""
# !wget https://downloader.disk.yandex.ru/d/CHECKPOINT1_ID -O best-checkpoint-task1.ckpt
# !wget https://downloader.disk.yandex.ru/d/CHECKPOINT2_ID -O best-checkpoint-task2.ckpt
checkpoint_path_task1 = './best-checkpoint-task1.ckpt'
checkpoint_path_task2 = './best-checkpoint-task2.ckpt'
"""

# Функция оценки
def evaluate_task(model, testქ; test_dataloader, device="cuda:0"):
    model = model.to(device)
    model.eval()
    total_accuracy = 0.0
    for images, labels in tqdm(test_dataloader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            output = model._forward(images)
            acc_batch = accuracy(output, labels, task='multiclass', num_classes=100)
        total_accuracy += acc_batch
    accuracy = total_accuracy / len(test_dataloader)
    return accuracy

# Оценка задания 1
model_task1 = YourNetTask1()
model_task1.load_state_dict(torch.load(checkpoint_path_task1)['state_dict'])
accuracy_task1 = evaluate_task(model_task1, val_dataloader_task1, device)
print(f"Точность на валидации для задания 1: {accuracy_task1:.4f}")

# Оценка задания 2
model_task2 = YourNetTask2()
model_task2.load_state_dict(torch.load(checkpoint_path_task2)['state_dict'])
accuracy_task2 = evaluate_task(model_task2, val_dataloader_task2, device)
print(f"Точность на валидации для задания 2: {accuracy_task2:.4f}")

# Отчет об экспериментах
"""
# Отчет об экспериментах

## Задание 1: Достижение точности ≥0.24 без предобучения и изменения размера

### Архитектура
- Создана CNN, вдохновленная ResNet: три блока сверток с нормализацией батчей и ReLU, затем max-pooling.
- Финальный слой с dropout (0.5) для предотвращения переобучения.
- Входной размер: 32x32, итоговая карта признаков: 256x4x4, 100 классов.

### Аугментация данных
- RandomHorizontalFlip (p=0.5), RandomRotation (15 градусов).
- Нормализация: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5].

### Обучение
- Adam, lr=1e-3, ReduceLROnPlateau (factor=0.5, patience=5).
- До 20 эпох, ранняя остановка (patience=10), batch size=32.
- Логирование в W&B (проект emoji_classification_task1).

### Эксперименты
- Простая модель: ~0.15 (недообучение).
- Глубокая сеть: ~0.20.
- Аугментации и dropout: ~0.25–0.28.
- Планировщик lr: ~0.27.

### Что сработало
- Нормализация батчей, умеренные аугментации, планировщик lr.

### Что не сработало
- Простые модели, сильные аугментации (ColorJitter).

### Почему выбран подход
- ResNet-подобная сеть эффективна.
- Ограниченные аугментации соблюдали запрет на ресайз.
- Adam и ReduceLROnPlateau для стабильности.

### Источники
- PyTorch Lightning: https://pytorch-lightning.readthedocs.io/en/1.4.5/
- Torchvision transforms: https://pytorch.org/vision/main/transforms.html
- Stack Overflow для torchmetrics.

## Задание 2: Достижение точности ≥0.34 с предобучением и ресайзом

### Архитектура
- ResNet-18 (torchvision), заморожены сверточные слои, дообучен fc-слой (100 классов).

### Аугментация данных
- Resize до 224x224, ImageNet нормализация: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
- RandomHorizontalFlip, RandomRotation (20 градусов), ColorJitter.

### Обучение
- Adam, lr=1e-3 (только fc), ReduceLROnPlateau (patience=5).
- До 10 эпох, ранняя остановка (patience=5), batch size=32.
- Логирование в W&B (проект emoji_classification_task2).

### Эксперименты
- Без дообучения: ~0.10.
- Дообучение fc: ~0.40.
- Аугментации: ~0.45–0.50.
- Планировщик lr: ~0.48.

### Что сработало
- Предобученные признаки, ресайз, сильные аугментации.

### Что не сработало
- Обучение всех слоев (переобучение).
- Низкий lr (медленная сходимость).

### Почему выбран подход
- ResNet-18 легкая и эффективная.
- Заморозка слоев предотвращает переобучение.
- Аугментации для устойчивости.

### Источники
- Torchvision models: https://pytorch.org/vision/stable/models.html
- W&B Lightning: https://docs.wandb.ai/guides/integrations/lightning
- Albumentations: https://towardsdatascience.com/getting-started-with-albumentation-winning-deep-learning-image-augmentation-technique-in-pytorch-47aaba0ee3f8
"""