# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from resnet import ResNet18
from mobilenet import MobileNetV2Model

# 创建目录保存模型
model_dir = "./models/mobile_net_v2"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 定义超参数
num_classes = 10  # MNIST数据集有10个类别
num_epochs = 10
batch_size = 128
learning_rate = 0.001

# 数据预处理
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 将图像调整到ResNet18所需的尺寸
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # 对灰度图像进行标准化
    ]
)

# 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)

# 初始化模型、损失函数和优化器
# model = ResNet18()
model = MobileNetV2Model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 初始化TensorBoard
writer = SummaryWriter(log_dir="./logs")

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}"
            )

    # 计算平均训练损失
    train_loss /= total_step

    # 验证模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # 计算平均验证损失
    val_loss /= len(test_loader)

    # 将损失写入TensorBoard
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)

    # 保存模型参数
    torch.save(
        model.state_dict(), os.path.join(model_dir, f"mobilenetv2_epoch_{epoch+1}.pth")
    )

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
    )

# 关闭TensorBoard
writer.close()

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(
    f"Test Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%"
)
