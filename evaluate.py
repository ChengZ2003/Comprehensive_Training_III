import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
from resnet import ResNet18
import os

# 定义图像预处理
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


# 定义评估函数
def evaluate(data_dir, model_path, device):
    correct = 0
    total = 0

    # 加载模型
    model = ResNet18()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 遍历数据目录中的所有图像
    for label in range(10):
        label_dir = os.path.join(data_dir, str(label))
        if not os.path.isdir(label_dir):
            continue

        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)

            # 加载图像
            image = Image.open(image_path).convert("L")
            image = transform(image)
            image = image.unsqueeze(0)  # 增加batch维度

            # 预测
            with torch.no_grad():
                image = image.to(device)
                outputs = model(image)
                _, predicted = torch.max(outputs.data, 1)
                total += 1
                correct += predicted.item() == int(label)

    # 计算准确率
    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MNIST model")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="path to the test dataset",
        default="./data/validation",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="path to the trained model",
        default="./models/resnet18_epoch_3.pth",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    accuracy = evaluate(args.data, args.model, device)
    print(f"Accuracy: {accuracy * 100:.2f}%")
