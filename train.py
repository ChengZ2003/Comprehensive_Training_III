import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from lenet import LeNet
from vgg import VGG
from resnet import ResNet18
from mobilenet import MobileNetV2
from tqdm import tqdm
import argparse
from config import (
    NUM_CLASSES,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
)


def evaluate_model(model, device, test_loader):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算以节省内存并加速评估
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 累加正确预测的数量

    accuracy = 100 * correct / total  # 计算准确率
    return accuracy


# 训练模型
def train_model(
    model_name,
    model,
    num_epochs,
    device,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    model_dir,
    writer,
    loss_log_file,
):
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        with tqdm(
            total=total_step,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            ncols=100,
            leave=False,
        ) as pbar:
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

                # 计算训练准确率
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                pbar.update(1)

        # 计算平均训练损失和准确率
        train_loss /= total_step
        train_accuracy = 100 * correct_train / total_train

        # 验证模型
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # 计算验证准确率
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        if i % 100 == 99:
            # 计算平均验证损失和准确率
            test_loss /= len(test_loader)
            test_accuracy = 100 * correct_test / total_test

            # 将损失和准确率写入TensorBoard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("Accuracy/train", train_accuracy, epoch)
            writer.add_scalar("Accuracy/test", test_accuracy, epoch)

        # 在每个epoch结束时保存模型和日志的部分需要调整，确保使用正确的模型名称
        torch.save(
            model.state_dict(), os.path.join(model_dir, f"{model_name}_{epoch+1}.pth")
        )

        # 将损失和准确率写入文件
        with open(loss_log_file, "a") as f:
            f.write(
                f"{epoch+1},{train_loss},{test_loss},{train_accuracy},{test_accuracy}\n"
            )

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, test Loss: {test_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, test Accuracy: {test_accuracy:.2f}%"
        )

    return model


if __name__ == "__main__":

    # 定义超参数
    num_classes = NUM_CLASSES  # MNIST数据集有10个类别
    num_epochs = NUM_EPOCHS
    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE

    parser = argparse.ArgumentParser(description="MNIST Model Training")
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet18", "mobilenetv2", "vgg16", "lenet"],
        default="resnet18",
        help="Model architecture to train (resnet18 or mobilenetv2 or vgg16 or lenet)",
    )
    args = parser.parse_args()
    if args.model == "resnet18":
        model = ResNet18(num_classes=num_classes)
    elif args.model == "mobilenetv2":
        model = MobileNetV2(num_classes=num_classes)
    elif args.model == "vgg16":
        model = VGG(num_classes=num_classes)
    elif args.model == "lenet":
        model = LeNet(num_classes=num_classes)
    else:
        raise ValueError("Invalid model architecture specified.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_dir = f"./models/{args.model}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 初始化TensorBoard
    log_dir = os.path.join("./logs", args.model)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=f"./logs/{args.model}")

    # 创建损失记录文件
    loss_log_file = os.path.join(model_dir, f"{args.model}_loss_log.txt")
    with open(loss_log_file, "w") as f:
        f.write("epoch,train_loss,test_loss,train_accuracy,test_accuracy\n")

    if args.model != "lenet":
        # 数据预处理
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # 将图像调整到ResNet18所需的尺寸
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # 对灰度图像进行标准化
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
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

    train_model(
        model_name=args.model,
        model=model,
        num_epochs=num_epochs,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        model_dir=model_dir,
        writer=writer,
        loss_log_file=loss_log_file,
    )

    test_accuracy = evaluate_model(model, device, test_loader)
    print(f"Test Accuracy of the model on the 10000 test images: {test_accuracy:.2f}%")
