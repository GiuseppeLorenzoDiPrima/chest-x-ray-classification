"""
Architettura ResNet con connessioni residue.

Supporta sia classificazione binaria (2 classi: NORMAL / PNEUMONIA)
che ternaria (3 classi: BACTERIA / NORMAL / VIRUS).
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Blocco residuo
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


# ---------------------------------------------------------------------------
# Architettura ResNet
# ---------------------------------------------------------------------------

class ResNet(nn.Module):

    def __init__(
        self,
        block,
        layers: list[int],
        classification_type: str,
        stride_size: list[int],
        padding_size: list[int],
        kernel_size: list[int],
        channels_of_color: int,
        planes: list[int],
        in_features: int,
        inplanes: int,
    ):
        super().__init__()
        num_classes = 2 if classification_type.lower() == "binary" else 3
        self.inplanes = inplanes

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels_of_color, self.inplanes,
                      kernel_size=kernel_size[0],
                      stride=stride_size[0],
                      padding=padding_size[0]),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=kernel_size[1],
            stride=stride_size[0],
            padding=padding_size[1],
        )

        self.layer = nn.ModuleList()
        self.layer.append(self._make_layer(block, planes[0], layers[0], stride_size[1]))
        for i in range(1, len(layers)):
            self.layer.append(self._make_layer(block, planes[i], layers[i], stride_size[0]))

        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size[0], stride=stride_size[1])
        self.fc = nn.Linear(in_features, num_classes)

        logger.info(
            f"ResNet costruita: {len(layers)} layer group, "
            f"{num_classes} classi ({classification_type})."
        )

    def _make_layer(self, block, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        for layer in self.layer:
            x = layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_resnet(config: dict) -> ResNet:
    """Costruisce una ResNet a partire dalla sezione 'resnet' del config."""
    cfg = config["resnet"]
    cls_type = config["classification"]["type"]
    model = ResNet(
        block=ResidualBlock,
        layers=cfg["layers"],
        classification_type=cls_type,
        stride_size=cfg["stride"],
        padding_size=cfg["padding"],
        kernel_size=cfg["kernel"],
        channels_of_color=cfg["channels"],
        planes=cfg["planes"],
        in_features=cfg["in_features"],
        inplanes=cfg["inplanes"],
    )
    return model
