import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()

        # 确保首层处理单通道输入
        self.block_1 = self._make_layer(64, 2, in_channels=1)
        self.block_2 = self._make_layer(
            128, 2, in_channels=64
        )  # 这里应该是64，来自上一层的输出通道数
        self.block_3 = self._make_layer(256, 3, in_channels=128)
        self.block_4 = self._make_layer(512, 3, in_channels=256)
        self.block_5 = self._make_layer(
            512, 3, in_channels=512, pool=False
        )  # 最后一层不进行池化
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # 自适应池化层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 使用flatten代替view，以支持动态尺寸
        x = self.classifier(x)
        return x

    def _make_layer(self, channels, convs, in_channels, pool=True):
        layers = []
        for _ in range(convs):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = channels  # 更新输入通道数为当前层的输出通道数
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)


# # 示例使用
# if __name__ == "__main__":
#     model = VGG(num_classes=10)
#     print(model)
