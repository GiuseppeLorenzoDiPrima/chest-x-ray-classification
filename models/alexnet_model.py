"""
Architettura AlexNet per classificazione di immagini.

Supporta sia classificazione binaria (2 classi: NORMAL / PNEUMONIA)
che ternaria (3 classi: BACTERIA / NORMAL / VIRUS).
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Architettura AlexNet
# ---------------------------------------------------------------------------

class AlexNet(nn.Module):

    def __init__(
        self,
        classification_type: str,
        stride_size: list[int],
        padding_size: list[int],
        kernel_size: list[int],
        channels_of_color: int,
        inplace: bool,
    ):
        super().__init__()
        num_classes = 2 if classification_type.lower() == "binary" else 3

        self.features = nn.Sequential(
            nn.Conv2d(channels_of_color, 96,
                      kernel_size=kernel_size[0],
                      stride=stride_size[0],
                      padding=padding_size[0]),
            nn.ReLU(inplace=inplace),
            nn.MaxPool2d(kernel_size=kernel_size[2], stride=stride_size[1]),

            nn.Conv2d(96, 256,
                      kernel_size=kernel_size[1],
                      padding=padding_size[0]),
            nn.ReLU(inplace=inplace),
            nn.MaxPool2d(kernel_size=kernel_size[2], stride=stride_size[1]),

            nn.Conv2d(256, 384,
                      kernel_size=kernel_size[2],
                      padding=padding_size[1]),
            nn.ReLU(inplace=inplace),

            nn.Conv2d(384, 384,
                      kernel_size=kernel_size[2],
                      padding=padding_size[1]),
            nn.ReLU(inplace=inplace),

            nn.Conv2d(384, 256,
                      kernel_size=kernel_size[2],
                      padding=padding_size[1]),
            nn.ReLU(inplace=inplace),
            nn.MaxPool2d(kernel_size=kernel_size[2], stride=stride_size[1]),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=inplace),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=inplace),
            nn.Linear(4096, num_classes),
        )

        logger.info(
            f"AlexNet costruita: {num_classes} classi ({classification_type})."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_alexnet(config: dict) -> AlexNet:
    """Costruisce una AlexNet a partire dalla sezione 'alexnet' del config."""
    cfg = config["alexnet"]
    cls_type = config["classification"]["type"]
    model = AlexNet(
        classification_type=cls_type,
        stride_size=cfg["stride"],
        padding_size=cfg["padding"],
        kernel_size=cfg["kernel"],
        channels_of_color=cfg["channels"],
        inplace=cfg["inplace"],
    )
    return model
