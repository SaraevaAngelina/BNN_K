"""
Шаг 1. Откройте https://colab.research.google.com
Шаг 2. Создайте новый блокнот: File → New Notebook
Шаг 3. (Опционально) Включите GPU: Runtime → Change runtime type → GPU
Шаг 4. В первой ячейке вставьте и запустите:
        !pip install -q torch torchvision matplotlib numpy
Шаг 5. Загрузите этот файл через боковую панель (иконка папки → Upload)
        ИЛИ скопируйте весь код в одну большую ячейку и запустите.
Шаг 6. Если загружали файл, в новой ячейке введите:
        exec(open('binary_nets_experiment.py').read())

ОПИСАНИЕ МОДЕЛЕЙ:
  1. BasicBinaryNet  — бинарные веса+активации, без BN, без skip-связей
  2. BNBinaryNet     — бинарные веса+активации + Batch Normalization
  3. BiRealNet       — бинарные веса+активации + BN + тождественные
                       обходные соединения (стиль Bi-Real Net)

ДИАГНОСТИЧЕСКИЕ ГРАФИКИ:
  • Кривые обучения: loss и accuracy (train/val)
  • Информационная плоскость: I(X;T) vs I(T;Y) по эпохам
  • Нормы градиентов по слоям (лог-шкала)
  • Распределения активаций до и после бинаризации
  • Итоговое сравнение точности и скорости сходимости
"""

# ══════════════════════════════════════════════════════════════════════
# 0. ИМПОРТЫ И ВОСПРОИЗВОДИМОСТЬ
# ══════════════════════════════════════════════════════════════════════
import os
import random
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "figure.dpi": 120,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Фиксируем все источники случайности ──────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Устройство: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════
# 1. ГИПЕРПАРАМЕТРЫ
# ══════════════════════════════════════════════════════════════════════
CFG = {
    "batch_size":  256,
    "epochs":       25,
    "lr":          1e-3,
    "hidden_dim":  512,
    "n_layers":      3,
    "n_classes":    10,
    "input_dim":   784,   # MNIST: 28×28
    "mi_samples": 3000,   # кол-во сэмплов для оценки MI
    "mi_every":      2,   # оценивать MI каждые N эпох
    "plot_dir":  "./plots",
}

os.makedirs(CFG["plot_dir"], exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# 2. ДАННЫЕ
# ══════════════════════════════════════════════════════════════════════
def get_loaders():
    """Загрузка MNIST с нормализацией и разбивкой train/val/test."""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_full = datasets.MNIST("./data", train=True,  download=True, transform=tf)
    test_ds    = datasets.MNIST("./data", train=False, download=True, transform=tf)

    n_val   = 5000
    n_train = len(train_full) - n_val
    train_ds, val_ds = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    kw = dict(num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=512,                shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=512,                shuffle=False, **kw)

    print(f"[INFO] MNIST: train={n_train} | val={n_val} | test={len(test_ds)}")
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════
# 3. БИНАРНЫЕ ПРИМИТИВЫ
# ══════════════════════════════════════════════════════════════════════
class _BinaryActivationSTE(torch.autograd.Function):
    """
    Функция знака с обратным распространением по методу
    Straight-Through Estimator (STE): градиент проходит там, где |x| ≤ 1.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        # Clipped STE
        return grad_out * (x.abs() <= 1).float()

binary_activation = _BinaryActivationSTE.apply


class BinaryLinear(nn.Linear):
    """
    Линейный слой с бинаризованными весами (+1/−1) на прямом ходе.
    Реальные вещественные веса хранятся для обновления по градиенту.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_bin = binary_activation(self.weight)   # веса → ±1
        return F.linear(x, w_bin, self.bias)


# ══════════════════════════════════════════════════════════════════════
# 4. АРХИТЕКТУРЫ МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════════

class BasicBinaryNet(nn.Module):
    """
    Базовая бинарная сеть без BatchNorm и без skip-соединений.
    Архитектура: Input→Linear(512)→[BinaryLinear→Sign]×3→Linear(10)
    """
    def __init__(self, input_dim=784, hidden_dim=512,
                 n_classes=10, n_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.bin_layers = nn.ModuleList([
            BinaryLinear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, n_classes)

        # Буферы для диагностики (заполняются при forward)
        self.pre_binary_acts:  list[torch.Tensor] = []
        self.post_binary_acts: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.input_proj(x)

        self.pre_binary_acts  = []
        self.post_binary_acts = []

        for layer in self.bin_layers:
            x = layer(x)
            self.pre_binary_acts.append(x.detach())
            x = binary_activation(x)
            self.post_binary_acts.append(x.detach())

        return self.classifier(x)


class BNBinaryNet(nn.Module):
    """
    Бинарная сеть с BatchNorm перед бинаризацией.
    BN нормализует распределение активаций → улучшает устойчивость знака.
    """
    def __init__(self, input_dim=784, hidden_dim=512,
                 n_classes=10, n_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.bin_layers = nn.ModuleList([
            BinaryLinear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, n_classes)

        self.pre_binary_acts:  list[torch.Tensor] = []
        self.post_binary_acts: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.input_proj(x)

        self.pre_binary_acts  = []
        self.post_binary_acts = []

        for layer, bn in zip(self.bin_layers, self.bns):
            x = bn(layer(x))                        # BN нормализует
            self.pre_binary_acts.append(x.detach())
            x = binary_activation(x)               # затем бинаризация
            self.post_binary_acts.append(x.detach())

        return self.classifier(x)


class _BiRealBlock(nn.Module):
    """
    Один блок Bi-Real Net:
        out = sign(BN(W_b · x)) + x   ← тождественное post-activation shortcut
    Shortcut добавляет x ПОСЛЕ знаковой функции, сохраняя непрерывный
    поток градиента через слой (x не бинаризуется повторно).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.linear = BinaryLinear(dim, dim)
        self.bn     = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre  = self.bn(self.linear(x))        # до бинаризации
        post = binary_activation(pre) + x     # ±1 + вещественный shortcut
        # Сохраняем для диагностики
        self._pre_binary  = pre.detach()
        self._post_binary = post.detach()
        return post


class BiRealNet(nn.Module):
    """
    Бинарная сеть с тождественными обходными соединениями (Bi-Real Net).
    Skip-связи обеспечивают непрерывный поток информации и градиентов.
    """
    def __init__(self, input_dim=784, hidden_dim=512,
                 n_classes=10, n_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn   = nn.BatchNorm1d(hidden_dim)
        self.blocks     = nn.ModuleList([
            _BiRealBlock(hidden_dim) for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, n_classes)

    @property
    def pre_binary_acts(self) -> list[torch.Tensor]:
        return [b._pre_binary  for b in self.blocks if hasattr(b, "_pre_binary")]

    @property
    def post_binary_acts(self) -> list[torch.Tensor]:
        return [b._post_binary for b in self.blocks if hasattr(b, "_post_binary")]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        # Входная проекция + бинаризация (без skip, нет предыдущего слоя)
        x = binary_activation(self.input_bn(self.input_proj(x)))
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)


# ══════════════════════════════════════════════════════════════════════
# 5. ОЦЕНКА ВЗАИМНОЙ ИНФОРМАЦИИ (Информационная плоскость)
# ══════════════════════════════════════════════════════════════════════

def _marginal_entropy_binning(acts: np.ndarray, n_bins: int = 30) -> float:
    """
    Оценка H(T) через биннинг по каждому нейрону независимо (upper bound).
    H(T) ≤ Σ_i H(T_i)  — оценка сверху при маргинальной независимости.
    Возвращает среднее по нейронам (бит).
    """
    n, d = acts.shape
    total_h = 0.0
    vmin, vmax = acts.min(axis=0), acts.max(axis=0)

    for i in range(d):
        lo, hi = vmin[i], vmax[i]
        if hi - lo < 1e-9:          # нейрон «мёртв»
            continue
        hist, _ = np.histogram(acts[:, i], bins=n_bins, range=(lo, hi))
        p = hist / (hist.sum() + 1e-12)
        p = p[p > 0]
        total_h += -np.sum(p * np.log2(p))

    return total_h / d              # среднее по нейронам


def compute_mi_layer(
    activations: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 30,
    n_classes: int = 10,
) -> tuple[float, float]:
    """
    Оценка I(X;T) и I(T;Y) для одного слоя методом биннинга.

    Формулы:
        I(X;T) ≈ H(T)              (для детерминированной сети H(T|X)=0)
        I(T;Y) = H(T) - H(T|Y)
        H(T|Y) = Σ_y P(y)·H(T|Y=y)

    Параметры биннинга одинаковы для обоих путей → оценки согласованны.
    """
    acts_np   = activations.cpu().float().numpy()
    labels_np = labels.cpu().numpy()

    # ── I(X;T) ─────────────────────────────────────────────
    ht = _marginal_entropy_binning(acts_np, n_bins)
    ixt = ht

    # ── I(T;Y) ─────────────────────────────────────────────
    ht_given_y = 0.0
    for y in range(n_classes):
        mask = labels_np == y
        if mask.sum() < 5:
            continue
        py = mask.mean()
        ht_y = _marginal_entropy_binning(acts_np[mask], n_bins)
        ht_given_y += py * ht_y

    ity = max(0.0, ht - ht_given_y)
    return float(ixt), float(ity)


@torch.no_grad()
def collect_layer_activations(
    model: nn.Module,
    loader: DataLoader,
    n_samples: int = 3000,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Прогон данных через модель для сбора активаций всех слоёв."""
    model.eval()
    all_acts_per_layer: list[list[torch.Tensor]] = []
    all_labels: list[torch.Tensor] = []
    collected = 0

    for x, y in loader:
        if collected >= n_samples:
            break
        _ = model(x.to(DEVICE))

        if collected == 0:
            all_acts_per_layer = [[] for _ in range(len(model.post_binary_acts))]

        for i, act in enumerate(model.post_binary_acts):
            all_acts_per_layer[i].append(act.cpu())
        all_labels.append(y)
        collected += x.size(0)

    labels = torch.cat(all_labels)[:n_samples]
    acts   = [torch.cat(a)[:n_samples] for a in all_acts_per_layer]
    return acts, labels


def compute_mi_trajectory(
    model: nn.Module,
    loader: DataLoader,
    n_samples: int = 3000,
    n_classes: int = 10,
) -> list[tuple[float, float]]:
    """Вернуть список (I(X;T), I(T;Y)) для каждого слоя."""
    acts_list, labels = collect_layer_activations(model, loader, n_samples)
    return [compute_mi_layer(a, labels, n_classes=n_classes) for a in acts_list]


# ══════════════════════════════════════════════════════════════════════
# 6. ЦИКЛ ОБУЧЕНИЯ
# ══════════════════════════════════════════════════════════════════════

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.train()
    total_loss = total_correct = total = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total         += x.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = total_correct = total = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss   = criterion(logits, y)

        total_loss    += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total         += x.size(0)

    return total_loss / total, total_correct / total


def _get_grad_norms(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> dict[str, float]:
    """Вычислить нормы градиентов одним мини-батчем (не меняет параметры)."""
    model.train()
    x, y = next(iter(loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    loss = criterion(model(x), y)
    loss.backward()
    norms = {
        name: param.grad.norm().item()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    # Очищаем градиенты, чтобы не влиять на следующий шаг
    for p in model.parameters():
        p.grad = None
    return norms


def run_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    name: str,
    epochs: int = 25,
) -> dict:
    """
    Полный цикл обучения с логированием всех диагностических метрик.
    Возвращает словарь `history`.
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history: dict = {
        "train_loss":   [],
        "val_loss":     [],
        "train_acc":    [],
        "val_acc":      [],
        "grad_norms":   [],          # список dict {layer_name: norm} по эпохам
        "mi_trajectory": [],         # список (epoch, [(ixt, ity), ...])
    }

    bar = "─" * 55
    print(f"\n{bar}")
    print(f"  Обучение: {name}")
    print(bar)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc     = _eval_epoch(model, val_loader, criterion)
        scheduler.step()

        # Нормы градиентов
        grad_norms = _get_grad_norms(model, train_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["grad_norms"].append(grad_norms)

        # Взаимная информация (дорогостоящая операция — реже)
        if epoch % CFG["mi_every"] == 0 or epoch == 1:
            mi = compute_mi_trajectory(
                model, val_loader,
                n_samples=CFG["mi_samples"],
                n_classes=CFG["n_classes"],
            )
            history["mi_trajectory"].append((epoch, mi))

        print(
            f"  Эпоха {epoch:3d}/{epochs} │ "
            f"Train loss={train_loss:.4f} acc={train_acc:.3f} │ "
            f"Val   loss={val_loss:.4f} acc={val_acc:.3f}"
        )

    return history


# ══════════════════════════════════════════════════════════════════════
# 7. ВИЗУАЛИЗАЦИЯ
# ══════════════════════════════════════════════════════════════════════
PALETTE = {
    "BasicBinaryNet": "#E74C3C",
    "BNBinaryNet":    "#2ECC71",
    "BiRealNet":      "#3498DB",
}
LAYER_PALETTE = ["#FF6B6B", "#FFA500", "#4ECDC4"]


def _save(fig: plt.Figure, fname: str) -> None:
    path = os.path.join(CFG["plot_dir"], fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Сохранено: {path}")


# ── 7.1 Кривые обучения ──────────────────────────────────────────────
def plot_training_curves(histories: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Кривые обучения и валидации", fontsize=15, fontweight="bold", y=1.01)

    slots = [
        (0, 0, "train_loss",  "Train Loss"),
        (0, 1, "train_acc",   "Train Accuracy"),
        (1, 0, "val_loss",    "Val Loss"),
        (1, 1, "val_acc",     "Val Accuracy"),
    ]

    for row, col, key, title in slots:
        ax = axes[row, col]
        for name, hist in histories.items():
            epochs = range(1, len(hist[key]) + 1)
            ax.plot(epochs, hist[key], color=PALETTE[name],
                    linewidth=2.2, label=name, marker="o",
                    markersize=3.5, markevery=5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Эпоха")
        ax.set_ylabel("Loss" if "loss" in key else "Accuracy")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25, linestyle="--")
        if "acc" in key:
            ax.set_ylim(0, 1.05)

    plt.tight_layout()
    _save(fig, "01_training_curves.png")
    plt.show()


# ── 7.2 Информационная плоскость ─────────────────────────────────────
def plot_information_plane(histories: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle(
        "Информационная плоскость: траектории по слоям и эпохам",
        fontsize=14, fontweight="bold",
    )

    for ax, (name, hist) in zip(axes, histories.items()):
        traj = hist["mi_trajectory"]
        if not traj:
            continue

        n_layers = len(traj[0][1])
        epochs_recorded = [e for e, _ in traj]

        for li in range(n_layers):
            c     = LAYER_PALETTE[li % len(LAYER_PALETTE)]
            ixts  = [mi[li][0] for _, mi in traj]
            itys  = [mi[li][1] for _, mi in traj]

            # Траектория
            ax.plot(ixts, itys, color=c, alpha=0.55, linewidth=1.8, zorder=2)

            # Точки, окрашенные по эпохе
            sc = ax.scatter(
                ixts, itys,
                c=epochs_recorded,
                cmap="plasma",
                vmin=1, vmax=CFG["epochs"],
                s=70, zorder=4,
                edgecolors=c, linewidths=1.2,
                label=f"Слой {li + 1}",
            )

            # Маркеры начала и конца
            ax.annotate("●", (ixts[0],  itys[0]),  fontsize=11, color=c,
                        ha="center", va="center", zorder=5)
            ax.annotate("★", (ixts[-1], itys[-1]), fontsize=12, color=c,
                        ha="center", va="center", zorder=5)

        cb = plt.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("Эпоха", fontsize=9)

        ax.set_xlabel("I(X ; T) [бит]", fontsize=11)
        ax.set_ylabel("I(T ; Y) [бит]", fontsize=11)
        ax.set_title(name, fontsize=11, fontweight="bold",
                     color=PALETTE[name])
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.2, linestyle="--")

        # Аннотация направления
        ax.annotate(
            "● — старт    ★ — финал",
            xy=(0.02, 0.03), xycoords="axes fraction",
            fontsize=7.5, color="grey",
        )

    plt.tight_layout()
    _save(fig, "02_information_plane.png")
    plt.show()


# ── 7.3 Нормы градиентов ────────────────────────────────────────────
def plot_grad_norms(histories: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle("Нормы градиентов по слоям (log-шкала)",
                 fontsize=14, fontweight="bold")

    for ax, (name, hist) in zip(axes, histories.items()):
        grad_list = hist["grad_norms"]
        if not grad_list:
            continue

        # Оставляем только слои с весами
        layer_names = [
            ln for ln in grad_list[0]
            if "weight" in ln and "classifier" not in ln
        ]
        epochs = range(1, len(grad_list) + 1)
        cmap   = cm.get_cmap("plasma", len(layer_names))

        for i, ln in enumerate(layer_names):
            norms = [g.get(ln, np.nan) for g in grad_list]
            label = (ln
                     .replace(".weight", "")
                     .replace("bin_layers.", "BinL")
                     .replace("blocks.", "Block")
                     .replace(".linear", ""))
            ax.semilogy(
                epochs, norms,
                color=cmap(i), linewidth=2.0,
                label=label, alpha=0.85,
            )

        ax.set_title(name, fontsize=11, fontweight="bold",
                     color=PALETTE[name])
        ax.set_xlabel("Эпоха")
        ax.set_ylabel("||∇W||₂")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.2, linestyle="--", which="both")

    plt.tight_layout()
    _save(fig, "03_grad_norms.png")
    plt.show()


# ── 7.4 Распределения активаций ──────────────────────────────────────
def plot_activation_distributions(
    models_dict: dict,
    val_loader: DataLoader,
) -> None:
    n_layers  = CFG["n_layers"]
    n_models  = len(models_dict)
    n_cols    = n_layers * 2          # пара pre/post для каждого слоя

    fig, axes = plt.subplots(
        n_models, n_cols,
        figsize=(n_cols * 3.5, n_models * 3.3),
    )
    if n_models == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Распределения активаций: до и после бинаризации",
        fontsize=14, fontweight="bold",
    )

    # Берём один батч данных
    x_batch, _ = next(iter(val_loader))
    x_batch = x_batch.to(DEVICE)

    for row, (name, model) in enumerate(models_dict.items()):
        model.eval()
        with torch.no_grad():
            _ = model(x_batch)

        pre_acts  = model.pre_binary_acts
        post_acts = model.post_binary_acts
        c = PALETTE[name]

        for li in range(min(n_layers, len(pre_acts))):
            # ── Pre-binarization ──
            ax_pre = axes[row, li * 2]
            pre_data = pre_acts[li].cpu().float().numpy().flatten()

            ax_pre.hist(pre_data, bins=60, color=c, alpha=0.75,
                        edgecolor="white", linewidth=0.2)
            ax_pre.axvline(0, color="red", linestyle="--",
                           linewidth=1.8, label="порог")
            ax_pre.set_title(
                f"{name}\nСлой {li+1} — ДО бинаризации",
                fontsize=8.5, fontweight="bold", color=c,
            )
            ax_pre.set_xlabel("Значение активации", fontsize=8)
            ax_pre.set_ylabel("Частота", fontsize=8)
            ax_pre.legend(fontsize=7.5)
            ax_pre.grid(alpha=0.2, linestyle="--")

            # Аннотация: доля >0 и <0
            frac_pos = (pre_data > 0).mean()
            ax_pre.annotate(
                f"+1: {frac_pos:.1%}   −1: {1-frac_pos:.1%}",
                xy=(0.03, 0.93), xycoords="axes fraction",
                fontsize=7.5, color="grey",
            )

            # ── Post-binarization ──
            ax_post = axes[row, li * 2 + 1]
            post_data = post_acts[li].cpu().float().numpy().flatten()

            unique, counts = np.unique(
                np.round(post_data, 1), return_counts=True
            )
            ax_post.bar(unique, counts, width=0.15, color=c,
                        alpha=0.85, edgecolor="white")
            ax_post.set_title(
                f"{name}\nСлой {li+1} — ПОСЛЕ бинаризации",
                fontsize=8.5, fontweight="bold", color=c,
            )
            ax_post.set_xlabel("Значение активации", fontsize=8)
            ax_post.set_ylabel("Частота", fontsize=8)
            ax_post.grid(alpha=0.2, linestyle="--", axis="y")

    plt.tight_layout(h_pad=2.5)
    _save(fig, "04_activation_distributions.png")
    plt.show()


# ── 7.5 Итоговое сравнение ───────────────────────────────────────────
def plot_summary(histories: dict) -> None:
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    names = list(histories.keys())
    colors = [PALETTE[n] for n in names]

    # ── (A) Финальная точность ──────────────────────────────
    ax_a = fig.add_subplot(gs[0])
    final_train = [histories[n]["train_acc"][-1] * 100 for n in names]
    final_val   = [histories[n]["val_acc"][-1]   * 100 for n in names]
    x = np.arange(len(names))
    w = 0.35

    b1 = ax_a.bar(x - w/2, final_train, w, color=colors, alpha=0.90, label="Train")
    b2 = ax_a.bar(x + w/2, final_val,   w, color=colors, alpha=0.45,
                  edgecolor="black", linewidth=0.8, label="Val")

    for b, v in zip(list(b1) + list(b2),
                    final_train + final_val):
        ax_a.text(b.get_x() + b.get_width()/2, v + 0.4,
                  f"{v:.1f}", ha="center", fontsize=9)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(
        [n.replace("Net", "\nNet") for n in names],
        fontsize=9,
    )
    ax_a.set_ylabel("Точность (%)")
    ax_a.set_title("Финальная точность", fontweight="bold")
    ax_a.set_ylim(0, 108)
    ax_a.legend(fontsize=9)
    ax_a.grid(axis="y", alpha=0.25, linestyle="--")

    # ── (B) Эпох до 85% val accuracy ────────────────────────
    ax_b = fig.add_subplot(gs[1])
    target = 0.85
    conv_epochs = []
    for n in names:
        reached = [i + 1 for i, a in enumerate(histories[n]["val_acc"])
                   if a >= target]
        conv_epochs.append(reached[0] if reached else CFG["epochs"])

    bars = ax_b.bar(
        names, conv_epochs,
        color=colors, alpha=0.85, edgecolor="white", linewidth=0.5,
    )
    for b, v in zip(bars, conv_epochs):
        ax_b.text(b.get_x() + b.get_width()/2, v + 0.3,
                  str(v), ha="center", fontsize=11, fontweight="bold")

    ax_b.set_xticklabels(
        [n.replace("Net", "\nNet") for n in names],
        fontsize=9,
    )
    ax_b.set_ylabel("Эпох")
    ax_b.set_title(f"Эпох до {target*100:.0f}% Val Acc", fontweight="bold")
    ax_b.grid(axis="y", alpha=0.25, linestyle="--")

    # ── (C) Финальный val loss ───────────────────────────────
    ax_c = fig.add_subplot(gs[2])
    final_val_loss = [histories[n]["val_loss"][-1] for n in names]
    bars2 = ax_c.bar(
        names, final_val_loss,
        color=colors, alpha=0.85, edgecolor="white", linewidth=0.5,
    )
    for b, v in zip(bars2, final_val_loss):
        ax_c.text(b.get_x() + b.get_width()/2, v + 0.003,
                  f"{v:.4f}", ha="center", fontsize=9)

    ax_c.set_xticklabels(
        [n.replace("Net", "\nNet") for n in names],
        fontsize=9,
    )
    ax_c.set_ylabel("Cross-Entropy Loss")
    ax_c.set_title("Финальный Val Loss", fontweight="bold")
    ax_c.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("Итоговое сравнение моделей", fontsize=14,
                 fontweight="bold", y=1.03)
    plt.tight_layout()
    _save(fig, "05_summary.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════
# 8. ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════════════

def main() -> tuple[dict, dict]:
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Бинарные нейронные сети — сравнительный эксперимент ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Устройство : {DEVICE}")
    print(f"  Эпох       : {CFG['epochs']}")
    print(f"  Batch size : {CFG['batch_size']}")
    print(f"  Hidden dim : {CFG['hidden_dim']}")

    # ── Загрузка данных ────────────────────────────────────
    train_loader, val_loader, test_loader = get_loaders()

    # ── Создание моделей ───────────────────────────────────
    model_constructors = {
        "BasicBinaryNet": BasicBinaryNet,
        "BNBinaryNet":    BNBinaryNet,
        "BiRealNet":      BiRealNet,
    }
    model_instances = {
        name: cls(
            CFG["input_dim"], CFG["hidden_dim"],
            CFG["n_classes"], CFG["n_layers"],
        )
        for name, cls in model_constructors.items()
    }

    print("\n  Параметры моделей:")
    for name, model in model_instances.items():
        n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    {name:<20}: {n_p:>9,} параметров")

    # ── Обучение ───────────────────────────────────────────
    histories: dict = {}
    trained_models: dict = {}

    for name, model in model_instances.items():
        hist = run_experiment(model, train_loader, val_loader,
                              name, epochs=CFG["epochs"])
        histories[name]      = hist
        trained_models[name] = model

    # ── Тест ──────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    print("\n╔══════════════════════════════════╗")
    print("║       Результаты на тесте        ║")
    print("╚══════════════════════════════════╝")
    for name, model in trained_models.items():
        tl, ta = _eval_epoch(model, test_loader, criterion)
        print(f"  {name:<20}: loss={tl:.4f}  acc={ta:.4f}  ({ta*100:.2f}%)")

    # ── Построение графиков ───────────────────────────────
    print(f"\n  Построение графиков → {CFG['plot_dir']}/")
    plot_training_curves(histories)
    plot_information_plane(histories)
    plot_grad_norms(histories)
    plot_activation_distributions(trained_models, val_loader)
    plot_summary(histories)

    print("\n  ✓ Эксперимент завершён. Все графики сохранены.")
    return histories, trained_models
if __name__ == "__main__":
    histories, models = main()
