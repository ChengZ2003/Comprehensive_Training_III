import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
from resnet import ResNet18
import os
import csv
from lenet import LeNet
from vgg import VGG
from mobilenet import MobileNetV2
from config import (
    NUM_CLASSES,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
)


def read_csv(file_path):
    data = []
    with open(file_path + "/label.csv", mode="r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            data.append({row[1]: row[0]})
    return data


# 定义评估函数
def evaluate(data, data_path, model, device, output_file):
    correct = 0
    total = len(data)

    for ele in data:
        file_name = list(ele.keys())[0].strip()
        label = int(list(ele.values())[0])
        image = Image.open(data_path + "/images/" + file_name).convert(
            "L"
        )  # MNIST数据集是灰度图像
        image = transform(image)
        image = image.unsqueeze(0)  # 增加batch维度

        with torch.no_grad():
            image = image.to(device)
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)

        # 写入csv文件
        with open(output_file, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([file_name, label, predicted.item()])

        if predicted.item() == label:
            correct += 1

    # 计算准确率
    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    num_classes = NUM_CLASSES  # MNIST数据集有10个类别
    num_epochs = NUM_EPOCHS
    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE

    parser = argparse.ArgumentParser(description="Evaluate MNIST model")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="path to the test dataset",
        default="./data/valid_set/",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["resnet18", "mobilenetv2", "vgg16", "lenet"],
        default="lenet",
        help="Model architecture to train (resnet18 or mobilenetv2 or vgg16 or lenet)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="path to the trained model",
        default="./models/resnet18_epoch_3.pth",
    )
    args = parser.parse_args()

    if args.model_name == "resnet18":
        model = ResNet18(num_classes=num_classes)
    elif args.model_name == "mobilenetv2":
        model = MobileNetV2(num_classes=num_classes)
    elif args.model_name == "vgg16":
        model = VGG(num_classes=num_classes)
    elif args.model_name == "lenet":
        model = LeNet(num_classes=num_classes)
    else:
        raise ValueError("Invalid model architecture specified.")

    if args.model_name != "lenet":
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
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    val_data = read_csv(args.data)
    device = torch.device("cpu")
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    output_file = "./data/valid_set/eval_result.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["file_name", "label", "predicted"])

    accuracy = evaluate(
        data=val_data,
        data_path=args.data,
        device=device,
        model=model,
        output_file=output_file,
    )
    print(f"Accuracy: {accuracy * 100:.2f}%")
