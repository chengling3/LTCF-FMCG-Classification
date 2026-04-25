import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
from datetime import datetime
import os

print(f"PyTorch 版本: {torch.__version__}")
print(f"torchvision 版本: {torchvision.__version__}")

# 创建保存模型和日志的目录
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# 数据预处理 - 训练集添加数据增强
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试集只需要进行必要的尺寸调整和归一化
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 加载数据集
def load_datasets(train_dir, test_dir, batch_size):
    trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transform)
    testset = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform)

    # 创建数据加载器
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"训练集大小: {len(trainset)} | 类别数: {len(trainset.classes)}")
    print(f"测试集大小: {len(testset)} | 类别数: {len(testset.classes)}")

    return trainloader, testloader, len(trainset.classes)


# 训练一个epoch
def train_epoch(model, trainloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    train_bar = tqdm(trainloader, desc=f'Epoch {epoch + 1}', leave=False)
    for i, (inputs, labels) in enumerate(train_bar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        train_bar.set_postfix({'loss': f'{running_loss / (i + 1):.4f}', 'acc': f'{100. * correct / total:.2f}%'})

    return running_loss / len(trainloader), 100. * correct / total


# 测试模型
def test_model(model, testloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        test_bar = tqdm(testloader, desc='Testing', leave=False)
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            test_bar.set_postfix(
                {'loss': f'{running_loss / (len(test_bar)):.4f}', 'acc': f'{100. * correct / total:.2f}%'})

    return running_loss / len(testloader), 100. * correct / total


# 主训练函数
def main():
    # 设置参数
    train_dir = r'/media/cp/TRAIN1/洗发水_小数据集_mini_两个都减少/train'
    test_dir = r'/media/cp/TRAIN1/洗发水_小数据集_mini_两个都减少/test'
    batch_size = 256
    epochs = 150
    lr = 0.01

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if str(device) == "cuda:0":
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")

    # 加载数据
    trainloader, testloader, num_classes = load_datasets(train_dir, test_dir, batch_size)

    # 初始化模型（使用旧版API加载预训练权重）
    model = torchvision.models.googlenet(pretrained=True)  # 旧版API
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    # 日志文件
    log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = open(f'logs/training_{log_time}.log', 'w')
    log_file.write(f"=== Training Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log_file.write(
        f"配置: Epochs={epochs}, BatchSize={batch_size}, LR={lr}, Optimizer=SGD, Scheduler=ReduceLROnPlateau\n")
    log_file.write(
        f"数据集: 训练集大小={len(trainloader.dataset)}, 测试集大小={len(testloader.dataset)}, 类别数={num_classes}\n")
    log_file.write("Epoch\tTrainLoss\tTrainAcc\tTestLoss\tTestAcc\tTime\tLR\n")

    # 最佳模型记录
    best_acc = 0.0
    best_epoch = 0
    best_model_path = f'models/best_model_{log_time}.pth'
    early_stopping_counter = 0
    early_stopping_patience = 15

    # 训练循环
    for epoch in range(epochs):
        start_time = time.time()

        # 训练和测试
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device, epoch)
        test_loss, test_acc = test_model(model, testloader, criterion, device)

        # 更新学习率
        scheduler.step(test_loss)

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 计算时间
        epoch_time = time.time() - start_time

        # 打印和记录日志
        log_entry = f"{epoch + 1}\t{train_loss:.4f}\t\t{train_acc:.2f}%\t\t{test_loss:.4f}\t\t{test_acc:.2f}%\t\t{epoch_time:.2f}s\t{current_lr:.6f}"
        print(log_entry)
        log_file.write(f"{log_entry}\n")
        log_file.flush()

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f'新的最佳模型已保存 - 准确率: {best_acc:.2f}%, 轮次: {best_epoch}')
            log_file.write(f'新的最佳模型已保存 - 准确率: {best_acc:.2f}%, 轮次: {best_epoch}\n')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f'未改进，最佳准确率仍为: {best_acc:.2f}% (轮次: {best_epoch})')
            log_file.write(f'未改进，最佳准确率仍为: {best_acc:.2f}% (轮次: {best_epoch})\n')

            # 早停检查
            if early_stopping_counter >= early_stopping_patience:
                print(f"早停触发: 在 {best_epoch} 轮后没有改进")
                log_file.write(f"早停触发: 在 {best_epoch} 轮后没有改进\n")
                break

    # 训练结束
    log_file.write(f"=== Training Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log_file.write(f"最佳模型准确率: {best_acc:.2f}% (轮次: {best_epoch})\n")
    log_file.close()

    print(f'训练完成，最佳模型已保存至: {best_model_path}')
    print(f'最佳准确率: {best_acc:.2f}% (轮次: {best_epoch})')


if __name__ == '__main__':
    from torch.multiprocessing import freeze_support

    freeze_support()
    main()
