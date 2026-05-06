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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def check_gradient_nan(model):
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            logging.warning(f"NaN detected in gradient of {name}")
            return True
    return False

def main():
    config = {
        'batch_size': 256,
        'learning_rate': 0.001,
        'rule_weights_lr': 0.0001,
        'weight_decay': 1e-5,
        'num_epochs': 150,
        'patience': 20,
        'factor': 0.5,
        'early_stopping': 20,
        'clip_grad_norm': 1.0,
        'data_dir': '/media/cp/data2/细粒度/paddleocr_RP_203/原始——paddleocr_RP_203/RP_203/',
        'similarity_dir': '/media/cp/data2/细粒度/paddleocr_RP_203/similarity_scores_四种/',
        'checkpoint_dir': './checkpoints',
        'best_model_path': 'best_model.pth',
        'resume': False,
        'resume_path': './checkpoints/latest_checkpoint.pth'
    }

    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    with open(os.path.join(config['checkpoint_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    logging.info(f"Training configuration: {config}")

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

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(train_dataset.labels)
    logging.info(f"Number of classes: {num_classes}")

    model = SimilarityFusionNet(num_classes=num_classes, pretrained=True)

    criterion = nn.CrossEntropyLoss()

    main_params = [p for n, p in model.named_parameters() if 'rule_weights' not in n]
    rule_params = [p for n, p in model.named_parameters() if 'rule_weights' in n]

    optimizer = optim.AdamW([
        {'params': main_params, 'lr': config['learning_rate']},
        {'params': rule_params, 'lr': config['rule_weights_lr']}
    ], weight_decay=config['weight_decay'])

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['factor'],
        patience=config['patience'],
        verbose=True,
        threshold=0.0001,
        threshold_mode='abs'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"Using device: {device}")

    start_epoch = 0
    best_val_acc = 0.0
    epochs_without_improvement = 0

    if config['resume'] and os.path.exists(config['resume_path']):
        checkpoint = torch.load(config['resume_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        epochs_without_improvement = checkpoint['epochs_without_improvement']
        logging.info(f"Resuming training from epoch {start_epoch} with best val acc: {best_val_acc:.2f}%")

    for epoch in range(start_epoch, config['num_epochs']):
        logging.info(f"{'=' * 20} Epoch {epoch + 1}/{config['num_epochs']} {'=' * 20}")

        with torch.no_grad():
            rule_weights = model.normalized_rule_weights
            logging.info(f"Current rule weights: {rule_weights.cpu().numpy().round(4)}")

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["num_epochs"]} Training', unit='batch')
        for batch_idx, (images, labels, sim_scores) in enumerate(train_pbar):
            images, labels, sim_scores = images.to(device), labels.to(device), sim_scores.to(device)

            optimizer.zero_grad()

            outputs = model(images, sim_scores)
            loss = criterion(outputs, labels)

            loss.backward()

            if config['clip_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])

            if check_gradient_nan(model):
                logging.warning("Gradient contains NaN! Skipping update...")
                optimizer.zero_grad()
                continue

            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            train_loss = running_loss / (batch_idx + 1)
            train_acc = 100. * correct / total
            train_pbar.set_postfix({'Loss': f'{train_loss:.4f}', 'Train Acc': f'{train_acc:.2f}%'})

        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = 100. * correct / total
        logging.info(f"Training Loss: {avg_train_loss:.4f}, Training Acc: {avg_train_acc:.2f}%")

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

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

                val_loss_avg = val_loss / (len(val_loader))
                val_acc = 100. * val_correct / val_total
                val_pbar.set_postfix({'Val Loss': f'{val_loss_avg:.4f}', 'Val Acc': f'{val_acc:.2f}%'})

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100. * val_correct / val_total
        logging.info(f"Validation Loss: {avg_val_loss:.4f}, Validation Acc: {avg_val_acc:.2f}%")

        scheduler.step(avg_val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        rule_lr = optimizer.param_groups[1]['lr']
        logging.info(f"Current learning rates - Main: {current_lr}, Rule weights: {rule_lr}")

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), config['best_model_path'])
            logging.info(f'Best model updated at Epoch {epoch + 1} with acc: {best_val_acc:.2f}%')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            logging.info(f'Epochs without improvement: {epochs_without_improvement}/{config["early_stopping"]}')

        if epochs_without_improvement >= config['early_stopping']:
            logging.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

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
    with torch.no_grad():
        final_rule_weights = model.normalized_rule_weights
        logging.info(f"Final rule weights: {final_rule_weights.cpu().numpy().round(4)}")

if __name__ == '__main__':
    main()