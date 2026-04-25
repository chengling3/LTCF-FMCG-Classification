import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torch.utils.data import DataLoader
from model import SimilarityFusionNet
from datasets import BatchDataset
from tqdm import tqdm
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)


# 设置随机种子，保证结果可复现
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# 梯度检查
def check_gradient_nan(model):
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            logging.warning(f"NaN detected in gradient of {name}")
            return True
    return False


def main():
    # 配置参数
    config = {
        'batch_size': 256,
        'learning_rate': 0.001,
        'rule_weights_lr': 0.0001,  # 规则权重的单独学习率
        'weight_decay': 1e-5,
        'num_epochs': 150,
        'patience': 20,  # 学习率调整耐心值
        'factor': 0.5,  # 学习率衰减因子
        'early_stopping': 20,  # 早停耐心值
        'clip_grad_norm': 1.0,  # 梯度裁剪范数
        'data_dir': '/media/cp/data2/细粒度/paddleocr_RP_203/原始——paddleocr_RP_203/RP_203/',
        'similarity_dir': '/media/cp/data2/细粒度/paddleocr_RP_203/similarity_scores_四种/',
        'checkpoint_dir': './checkpoints',
        'best_model_path': 'best_model.pth',
        'resume': False,  # 是否从检查点恢复训练
        'resume_path': './checkpoints/latest_checkpoint.pth'
    }

    # 创建检查点目录
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # 保存配置
    with open(os.path.join(config['checkpoint_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    logging.info(f"Training configuration: {config}")

    # 定义数据转换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = BatchDataset(
        root_dir=os.path.join(config['data_dir'], 'train'),
        label_file='class_indices.json',
        similarity_dir=os.path.join(config['similarity_dir'], 'train'),
        transform=train_transform,
        is_json=True
    )

    val_dataset = BatchDataset(
        root_dir=os.path.join(config['data_dir'], 'test'),
        label_file='class_indices.json',
        similarity_dir=os.path.join(config['similarity_dir'], 'test'),
        transform=val_transform,
        is_json=True
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # 获取类别数
    num_classes = len(train_dataset.labels)
    logging.info(f"Number of classes: {num_classes}")

    # 创建模型
    model = SimilarityFusionNet(num_classes=num_classes, pretrained=True)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 分离参数：主参数和规则权重参数
    main_params = [p for n, p in model.named_parameters() if 'rule_weights' not in n]
    rule_params = [p for n, p in model.named_parameters() if 'rule_weights' in n]

    optimizer = optim.AdamW([
        {'params': main_params, 'lr': config['learning_rate']},
        {'params': rule_params, 'lr': config['rule_weights_lr']}  # 规则权重使用单独学习率
    ], weight_decay=config['weight_decay'])

    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['factor'],
        patience=config['patience'],
        verbose=True,
        threshold=0.0001,
        threshold_mode='abs'
    )

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"Using device: {device}")

    # 初始化训练变量
    start_epoch = 0
    best_val_acc = 0.0
    epochs_without_improvement = 0

    # 从检查点恢复训练
    if config['resume'] and os.path.exists(config['resume_path']):
        checkpoint = torch.load(config['resume_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        epochs_without_improvement = checkpoint['epochs_without_improvement']
        logging.info(f"Resuming training from epoch {start_epoch} with best val acc: {best_val_acc:.2f}%")

    # 训练循环
    for epoch in range(start_epoch, config['num_epochs']):
        logging.info(f"{'=' * 20} Epoch {epoch + 1}/{config['num_epochs']} {'=' * 20}")

        # 打印当前规则权重
        with torch.no_grad():
            rule_weights = model.normalized_rule_weights
            logging.info(f"Current rule weights: {rule_weights.cpu().numpy().round(4)}")

        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 创建训练进度条
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["num_epochs"]} Training', unit='batch')
        for batch_idx, (images, labels, sim_scores) in enumerate(train_pbar):
            images, labels, sim_scores = images.to(device), labels.to(device), sim_scores.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images, sim_scores)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()

            # 梯度裁剪
            if config['clip_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])

            # 检查梯度
            if check_gradient_nan(model):
                logging.warning("Gradient contains NaN! Skipping update...")
                optimizer.zero_grad()
                continue

            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条信息
            train_loss = running_loss / (batch_idx + 1)
            train_acc = 100. * correct / total
            train_pbar.set_postfix({'Loss': f'{train_loss:.4f}', 'Train Acc': f'{train_acc:.2f}%'})

        # 计算平均训练指标
        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = 100. * correct / total
        logging.info(f"Training Loss: {avg_train_loss:.4f}, Training Acc: {avg_train_acc:.2f}%")

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        # 创建验证进度条
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{config["num_epochs"]} Validation', unit='batch')
        with torch.no_grad():
            for images, labels, sim_scores in val_pbar:
                images, labels, sim_scores = images.to(device), labels.to(device), sim_scores.to(device)

                outputs = model(images, sim_scores)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                # 更新进度条信息
                val_loss_avg = val_loss / (len(val_loader))
                val_acc = 100. * val_correct / val_total
                val_pbar.set_postfix({'Val Loss': f'{val_loss_avg:.4f}', 'Val Acc': f'{val_acc:.2f}%'})

        # 计算平均验证指标
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100. * val_correct / val_total
        logging.info(f"Validation Loss: {avg_val_loss:.4f}, Validation Acc: {avg_val_acc:.2f}%")

        # 更新学习率
        scheduler.step(avg_val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        rule_lr = optimizer.param_groups[1]['lr']
        logging.info(f"Current learning rates - Main: {current_lr}, Rule weights: {rule_lr}")

        # 保存最佳模型
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), config['best_model_path'])
            logging.info(f'Best model updated at Epoch {epoch + 1} with acc: {best_val_acc:.2f}%')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            logging.info(f'Epochs without improvement: {epochs_without_improvement}/{config["early_stopping"]}')

        # 早停检查
        if epochs_without_improvement >= config['early_stopping']:
            logging.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'epochs_without_improvement': epochs_without_improvement
        }
        torch.save(checkpoint, os.path.join(config['checkpoint_dir'], 'latest_checkpoint.pth'))
        logging.info(f"Checkpoint saved at epoch {epoch + 1}")

    logging.info('Training completed!')
    logging.info(f'Best validation accuracy: {best_val_acc:.2f}%')
    logging.info(f'Best model saved at: {config["best_model_path"]}')
    # 打印最终规则权重
    with torch.no_grad():
        final_rule_weights = model.normalized_rule_weights
        logging.info(f"Final rule weights: {final_rule_weights.cpu().numpy().round(4)}")


if __name__ == '__main__':
    main()