"""
垃圾分类图像分类 — 训练脚本
数据集: TrashNet (6类: glass, paper, cardboard, plastic, metal, trash)
模型: 自定义CNN + ResNet-18 迁移学习对比实验
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models, datasets
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
import json
from datetime import datetime
import os
import sys


# ============ 配置 ============
class Config:
    DATA_DIR = Path(__file__).parent / "data"
    # TrashNet 数据集 (通过 kagglehub 下载或本地路径)
    EXTRACT_DIR = DATA_DIR / "trashnet" / "Garbage classification" / "Garbage classification"
    OUTPUT_DIR = Path(__file__).parent / "outputs"
    MODEL_DIR = Path(__file__).parent / "models"

    CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    NUM_CLASSES = len(CLASSES)
    IMG_SIZE = 224
    BATCH_SIZE = 32
    BATCH_SIZE_CNN = 16          # CustomCNN 用较小 batch 避免 OOM
    NUM_WORKERS = 0
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def download_dataset():
    """下载 TrashNet 数据集 (优先本地缓存，否则 kagglehub 下载)"""
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    if Config.EXTRACT_DIR.exists():
        imgs = list(Config.EXTRACT_DIR.rglob("*.jpg"))
        if imgs:
            print(f"[OK] 数据集已存在: {Config.EXTRACT_DIR} ({len(imgs)} 张图像)", flush=True)
            return Config.EXTRACT_DIR

    print(f"[下载] 通过 kagglehub 获取 TrashNet 数据集...", flush=True)
    import kagglehub, shutil
    cache_path = Path(kagglehub.dataset_download("asdasdasasdas/garbage-classification"))
    # 数据集在缓存中的实际位置
    src = cache_path / "Garbage classification" / "Garbage classification"
    if src.exists():
        Config.EXTRACT_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(src), str(Config.EXTRACT_DIR))
        print(f"[OK] 复制完成: {Config.EXTRACT_DIR}", flush=True)
    else:
        print(f"[WARN] 未找到预期目录结构，使用: {cache_path}", flush=True)
        return cache_path
    return Config.EXTRACT_DIR


class SubsetWithTransform(Dataset):
    """带独立 transform 的 Dataset 子集包装器"""
    def __init__(self, subset, transform=None):
        self.dataset = subset.dataset
        self.indices = subset.indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


def load_data():
    """加载数据集并返回各个 DataLoader"""
    # ImageNet 标准化参数 (用于 ResNet 迁移学习)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # 简单标准化 (用于自定义 CNN)
    simple_mean = [0.5, 0.5, 0.5]
    simple_std = [0.5, 0.5, 0.5]

    # ResNet 训练集变换 (含数据增强)
    train_tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # ResNet 验证/测试集变换 (无增强)
    val_tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # 自定义CNN 训练变换 (含增强)
    train_simple_tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=simple_mean, std=simple_std)
    ])

    # 自定义CNN 验证/测试变换
    val_simple_tf = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=simple_mean, std=simple_std)
    ])

    # 加载完整数据集 (不加 transform，后续通过 SubsetWithTransform 处理)
    full_ds = datasets.ImageFolder(root=str(Config.EXTRACT_DIR))

    # 固定划分数据集
    total = len(full_ds)
    train_n = int(total * Config.TRAIN_RATIO)
    val_n = int(total * Config.VAL_RATIO)
    test_n = total - train_n - val_n

    gen = torch.Generator().manual_seed(Config.SEED)
    train_sub, val_sub, test_sub = random_split(
        full_ds, [train_n, val_n, test_n], generator=gen
    )

    # 包装为带 transform 的数据集
    train_ds = SubsetWithTransform(train_sub, train_tf)
    val_ds = SubsetWithTransform(val_sub, val_tf)
    test_ds = SubsetWithTransform(test_sub, val_tf)

    train_simple = SubsetWithTransform(train_sub, train_simple_tf)
    val_simple = SubsetWithTransform(val_sub, val_simple_tf)
    test_simple = SubsetWithTransform(test_sub, val_simple_tf)

    # DataLoader (ResNet 用 BATCH_SIZE, CustomCNN 用 BATCH_SIZE_CNN)
    kw = dict(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, pin_memory=True)
    kw_cnn = dict(batch_size=Config.BATCH_SIZE_CNN, num_workers=Config.NUM_WORKERS, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    train_s_loader = DataLoader(train_simple, shuffle=True, **kw_cnn)
    val_s_loader = DataLoader(val_simple, shuffle=False, **kw_cnn)
    test_s_loader = DataLoader(test_simple, shuffle=False, **kw_cnn)

    # 打印数据集统计
    print(f"\n{'='*50}", flush=True)
    print(f"TrashNet 数据集统计", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"总数: {total} | 训练: {train_n} | 验证: {val_n} | 测试: {test_n}", flush=True)
    print(f"类别: {full_ds.classes}", flush=True)
    for i, cls in enumerate(full_ds.classes):
        n = full_ds.targets.count(i)
        print(f"  [{i}] {cls:<12s}: {n} 张", flush=True)
    print(f"{'='*50}\n", flush=True)

    return (train_loader, val_loader, test_loader,
            train_s_loader, val_s_loader, test_s_loader)


# ============ 自定义 CNN ============
class CustomCNN(nn.Module):
    """自定义 CNN: Conv-BN-ReLU-Pool × 4 → FC × 2"""
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3×224×224 → 32×112×112
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 32×112×112 → 64×56×56
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 64×56×56 → 128×28×28
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4: 128×28×28 → 256×14×14
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


def get_resnet_model(num_classes=6):
    """ResNet-18 迁移学习: 冻结 backbone，替换分类头"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # 冻结全部 backbone
    for param in model.parameters():
        param.requires_grad = False
    # 替换分类头
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_feat, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model


# ============ 训练 & 评估 ============
def run_epoch(model, loader, criterion, optimizer, device, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss, correct, total = 0.0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            out = model(imgs)
            loss = criterion(out, lbls)
            if is_train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, pred = out.max(1)
        total += lbls.size(0)
        correct += pred.eq(lbls).sum().item()
    return total_loss / total, 100.0 * correct / total


def train_model(model, train_loader, val_loader, name, lr=None):
    if lr is None:
        lr = Config.LEARNING_RATE

    model = model.to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc, best_ep, wait = 0.0, 0, 0
    log_path = Config.OUTPUT_DIR / f'{name}_progress.json'

    print(f"\n{'='*60}", flush=True)
    print(f"训练 {name} | 设备: {Config.DEVICE} | lr={lr} | epochs={Config.EPOCHS}", flush=True)
    print(f"{'='*60}", flush=True)

    for ep in range(Config.EPOCHS):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, Config.DEVICE, True)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, optimizer, Config.DEVICE, False)
        scheduler.step(vl_loss)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        print(f"Epoch {ep+1:02d} | TrLoss: {tr_loss:.4f} TrAcc: {tr_acc:.2f}% | "
              f"VlLoss: {vl_loss:.4f} VlAcc: {vl_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}",
              flush=True)

        # 每个 epoch 后保存进度 (防止进程崩溃丢失数据)
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({'name': name, 'epoch': ep+1, 'best_acc': best_acc,
                       'best_epoch': best_ep, 'history': history}, f, ensure_ascii=False)

        if vl_acc > best_acc:
            best_acc, best_ep, wait = vl_acc, ep + 1, 0
            Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': ep, 'model_state_dict': model.state_dict(),
                'val_acc': vl_acc, 'history': history,
            }, Config.MODEL_DIR / f"{name}_best.pth")
            print(f"  -> 保存最佳模型 (acc={best_acc:.2f}%)", flush=True)
        else:
            wait += 1
            if wait >= 10:
                print(f"提前停止 (epoch {ep+1}, 最佳: {best_ep}, acc={best_acc:.2f}%)", flush=True)
                break

    print(f"[完成] {name} | 最佳验证准确率: {best_acc:.2f}% (epoch {best_ep})", flush=True)
    return history, best_acc


def evaluate(model, loader, name):
    model = model.to(Config.DEVICE)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(Config.DEVICE)
            preds.extend(model(imgs).argmax(1).cpu().tolist())
            labels.extend(lbls.tolist())

    acc = accuracy_score(labels, preds)
    print(f"\n{'='*60}")
    print(f"测试评估: {name} | 准确率: {acc*100:.2f}%")
    print(classification_report(labels, preds, target_names=Config.CLASSES, digits=4))

    # 混淆矩阵
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(Config.CLASSES, rotation=45, ha='right')
    ax.set_yticklabels(Config.CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — {name}')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = Config.OUTPUT_DIR / f'cm_{name}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[保存] 混淆矩阵: {path}")
    return preds, labels, acc


def plot_comparison(h1, h2, n1, n2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, yl in [(ax1, 'loss', 'Loss'), (ax2, 'acc', 'Accuracy (%)')]:
        ax.plot(h1[f'train_{key}'], label=f'{n1} Train', lw=1.5)
        ax.plot(h1[f'val_{key}'], label=f'{n1} Val', lw=1.5)
        ax.plot(h2[f'train_{key}'], label=f'{n2} Train', lw=1.5, ls='--')
        ax.plot(h2[f'val_{key}'], label=f'{n2} Val', lw=1.5, ls='--')
        ax.set_xlabel('Epoch'); ax.set_ylabel(yl); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle('模型训练对比', fontsize=14)
    plt.tight_layout()
    path = Config.OUTPUT_DIR / 'training_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[保存] 训练对比图: {path}")


# ============ 主流程 ============
def main():
    print(f"\n{'#'*60}", flush=True)
    print(f"垃圾分类图像分类 — CNN vs ResNet-18 迁移学习", flush=True)
    print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}", flush=True)
    print(f"设备: {Config.DEVICE}", flush=True)
    print(f"{'#'*60}", flush=True)

    set_seed(Config.SEED)
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 数据准备
    download_dataset()
    train_ldr, val_ldr, test_ldr, train_s, val_s, test_s = load_data()

    # 2. ResNet-18 迁移学习 (先训练, 更快更稳定)
    print(f"\n{'#'*60}\n# 实验一: ResNet-18 迁移学习\n{'#'*60}", flush=True)
    rn = get_resnet_model(Config.NUM_CLASSES)
    total_p = sum(p.numel() for p in rn.parameters())
    train_p = sum(p.numel() for p in rn.parameters() if p.requires_grad)
    print(f"参数量: {total_p:,} (可训练: {train_p:,})", flush=True)

    rn_hist, rn_val = train_model(rn, train_ldr, val_ldr, "ResNet18")

    # 3. 自定义 CNN
    print(f"\n{'#'*60}\n# 实验二: 自定义 CNN (从零训练)\n{'#'*60}", flush=True)
    cnn = CustomCNN(Config.NUM_CLASSES)
    n_params = sum(p.numel() for p in cnn.parameters())
    print(f"参数量: {n_params:,} (全可训练) | batch_size={Config.BATCH_SIZE_CNN}", flush=True)

    cnn_hist, cnn_val = train_model(cnn, train_s, val_s, "CustomCNN")

    # 4. 测试评估
    print(f"\n{'#'*60}\n# 测试集评估\n{'#'*60}", flush=True)

    # 加载最佳模型进行测试
    rn_best = get_resnet_model(Config.NUM_CLASSES)
    ckpt = torch.load(Config.MODEL_DIR / "ResNet18_best.pth",
                      map_location=Config.DEVICE, weights_only=True)
    rn_best.load_state_dict(ckpt['model_state_dict'])
    print(f"加载 ResNet18 最佳模型 (epoch {ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc',0):.2f}%)", flush=True)
    _, _, rn_test = evaluate(rn_best, test_ldr, "ResNet18")

    cnn_best = CustomCNN(Config.NUM_CLASSES)
    ckpt = torch.load(Config.MODEL_DIR / "CustomCNN_best.pth",
                      map_location=Config.DEVICE, weights_only=True)
    cnn_best.load_state_dict(ckpt['model_state_dict'])
    print(f"加载 CustomCNN 最佳模型 (epoch {ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc',0):.2f}%)", flush=True)
    _, _, cnn_test = evaluate(cnn_best, test_s, "CustomCNN")

    # 5. 对比
    print(f"\n{'='*60}", flush=True)
    print(f"模型对比总结", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'指标':<20} {'ResNet-18':>12} {'自定义CNN':>12}", flush=True)
    print(f"{'-'*44}", flush=True)
    print(f"{'验证准确率':<20} {rn_val:>11.2f}% {cnn_val:>11.2f}%", flush=True)
    print(f"{'测试准确率':<20} {rn_test*100:>11.2f}% {cnn_test*100:>11.2f}%", flush=True)

    plot_comparison(cnn_hist, rn_hist, "CustomCNN", "ResNet18")

    # 保存实验报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'pytorch': torch.__version__,
        'cuda': torch.cuda.is_available(),
        'device': str(Config.DEVICE),
        'dataset': 'TrashNet',
        'classes': Config.CLASSES,
        'ResNet18': {
            'val_acc': rn_val,
            'test_acc': float(rn_test * 100),
            'params': sum(p.numel() for p in get_resnet_model(6).parameters()),
            'trainable_params': sum(p.numel() for p in get_resnet_model(6).parameters() if p.requires_grad),
        },
        'CustomCNN': {
            'val_acc': cnn_val,
            'test_acc': float(cnn_test * 100),
            'params': sum(p.numel() for p in CustomCNN(6).parameters()),
        },
        'hyperparameters': {
            'epochs': Config.EPOCHS, 'batch_size_resnet': Config.BATCH_SIZE,
            'batch_size_cnn': Config.BATCH_SIZE_CNN,
            'learning_rate': Config.LEARNING_RATE, 'img_size': Config.IMG_SIZE,
        }
    }
    with open(Config.OUTPUT_DIR / 'experiment_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] 实验报告: {Config.OUTPUT_DIR / 'experiment_report.json'}", flush=True)
    print(f"\n{'#'*60}\n全部实验完成!\n{'#'*60}", flush=True)


if __name__ == '__main__':
    main()
