# predict.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import argparse
from resnet import ResNet18

# 定义图像预处理
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


# 定义预测函数
def predict(image_path, model_path, device):
    # 加载图像
    image = Image.open(image_path).convert("L")  # MNIST数据集是灰度图像
    image = transform(image)
    image = image.unsqueeze(0)  # 增加batch维度

    # 加载模型
    model = ResNet18()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 预测
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    return predicted.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict MNIST digit using ResNet18")
    parser.add_argument("--image", type=str, required=True, help="path to the image")
    parser.add_argument(
        "--model", type=str, required=True, help="path to the trained model"
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    prediction = predict(args.image, args.model, device)
    print(f"The predicted digit is: {prediction}")
