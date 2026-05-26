# Binary Neural Networks — Comparative Experiment

Эксперимент по сравнению бинарных нейронных сетей на датасете MNIST с использованием PyTorch.  
Проект исследует влияние Batch Normalization и residual/skip-соединений на обучение бинаризованных сетей.

---

## Возможности проекта

- Сравнение трёх архитектур бинарных сетей:
  - `BasicBinaryNet`
  - `BNBinaryNet`
  - `BiRealNet`
- Полностью воспроизводимый эксперимент
- Поддержка CPU и GPU
- Обучение на MNIST
- Визуальная диагностика процесса обучения
- Анализ информационной плоскости (`I(X;T)` vs `I(T;Y)`)
- Анализ распределений активаций
- Мониторинг норм градиентов
- Автоматическое сохранение графиков

---

# Архитектуры моделей

## 1. BasicBinaryNet

Базовая бинарная сеть:

- бинарные веса (`±1`)
- бинарные активации (`sign`)
- без BatchNorm
- без residual connections

Архитектура:

```text
Input
 → Linear
 → [BinaryLinear → Sign] × N
 → Linear
 → Output
```

---

## 2. BNBinaryNet

Улучшенная бинарная сеть:

- бинарные веса
- бинарные активации
- Batch Normalization перед бинаризацией

BatchNorm стабилизирует распределение активаций и улучшает прохождение градиентов.

---

## 3. BiRealNet

Сеть в стиле Bi-Real Net:

- бинарные веса
- бинарные активации
- BatchNorm
- residual/skip connections

Shortcut-соединения помогают:

- сохранять поток информации
- уменьшать деградацию градиентов
- ускорять сходимость

---

# Используемые методы

## Бинаризация

Используется функция знака:

```python
sign(x) → {-1, +1}
```

Для обратного распространения применяется:

### Straight-Through Estimator (STE)

Градиент пропускается только в области:

```text
|x| ≤ 1
```

Это позволяет обучать бинарные веса с помощью обычного backpropagation.

---

# Диагностика и визуализация

Во время эксперимента автоматически строятся графики:

## 1. Кривые обучения

- Train Loss
- Validation Loss
- Train Accuracy
- Validation Accuracy

Файл:

```text
plots/01_training_curves.png
```

---

## 2. Информационная плоскость

Траектории обучения слоёв:

```text
I(X;T) vs I(T;Y)
```

Позволяет анализировать:

- компрессию информации
- динамику представлений
- информативность признаков

Файл:

```text
plots/02_information_plane.png
```

---

## 3. Нормы градиентов

Показывает стабильность обучения бинарных сетей.

Используется логарифмическая шкала.

Файл:

```text
plots/03_grad_norms.png
```

---

## 4. Распределения активаций

До и после бинаризации:

- распределение вещественных активаций
- доля `+1` и `−1`
- влияние BatchNorm

Файл:

```text
plots/04_activation_distributions.png
```

---

## 5. Итоговое сравнение моделей

Сравнение:

- финальной accuracy
- скорости сходимости
- validation loss

Файл:

```text
plots/05_summary.png
```

---

# Структура проекта

```text
.
├── binary_nets_experiment.py
├── data/
├── plots/
│   ├── 01_training_curves.png
│   ├── 02_information_plane.png
│   ├── 03_grad_norms.png
│   ├── 04_activation_distributions.png
│   └── 05_summary.png
└── README.md
```

---

# Установка

## Требования

- Python 3.10+
- PyTorch
- torchvision
- numpy
- matplotlib

---

## Установка зависимостей

```bash
pip install torch torchvision matplotlib numpy
```

---

# Запуск

## Локально

```bash
python binary_nets_experiment.py
```

---

## Google Colab

### 1. Откройте Colab

https://colab.research.google.com

---

### 2. Установите зависимости

```python
!pip install -q torch torchvision matplotlib numpy
```

---

### 3. Загрузите файл

Загрузите `binary_nets_experiment.py` через боковую панель Colab.

---

### 4. Запустите

```python
exec(open('binary_nets_experiment.py').read())
```

---

# Гиперпараметры по умолчанию

```python
CFG = {
    "batch_size": 256,
    "epochs": 25,
    "lr": 1e-3,
    "hidden_dim": 512,
    "n_layers": 3,
    "n_classes": 10,
    "input_dim": 784,
}
```

---

# Воспроизводимость

Для полной воспроизводимости фиксируются:

- `random.seed`
- `numpy.random.seed`
- `torch.manual_seed`
- CUDA seeds
- deterministic cuDNN

Используемый seed:

```python
SEED = 42
```

---

# Результаты эксперимента

Ожидаемое поведение моделей:

| Модель | Особенности | Ожидаемый результат |
|---|---|---|
| BasicBinaryNet | Без BN и skip | Худшая стабильность |
| BNBinaryNet | BatchNorm | Лучше convergence |
| BiRealNet | BN + residual | Лучшая accuracy |

---

# Теоретическая мотивация

Бинарные сети позволяют:

- резко уменьшить потребление памяти
- ускорить inference
- заменить умножения на битовые операции
- запускать модели на edge-устройствах

Однако бинаризация создаёт проблемы:

- нестабильные градиенты
- потеря информации
- сложность оптимизации

Этот проект демонстрирует, как:

- BatchNorm
- residual connections
- STE

смягчают эти проблемы.
