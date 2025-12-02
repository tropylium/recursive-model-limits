import pydantic
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils import FeedForwardModel


class SimpleCNNConfig(pydantic.BaseModel):
    """
    Configuration for the SimpleCNN model, which performs image classification.
    """
    num_classes: int
    input_channels: int
    input_width: int  # same as height, assume square image
    num_conv_layers: int
    num_fc_layers: int  # at least two layers
    fc_hidden_dim: int
    dropout_rate: float = 0.2


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        max_pool: bool = False,
        maxpool_padding: int = 0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.SiLU(inplace=True)
        self.max_pool = (
            nn.MaxPool2d(kernel_size=2, stride=2, padding=maxpool_padding)
            if max_pool
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        if self.max_pool is not None:
            x = self.max_pool(x)
        return x


class SimpleCNN(FeedForwardModel):
    """
    Your standard CNN model.
    """

    def __init__(self, config: SimpleCNNConfig):
        super().__init__()
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()

        current_in_channels = config.input_channels
        current_out_channels = config.input_channels * 2
        conv_layers = []

        final_image_width = config.input_width
        for i in range(config.num_conv_layers):
            # Calculate padding: use padding=1 only if width is odd
            maxpool_padding = 1 if final_image_width % 2 == 1 else 0

            conv_layers.append(
                ConvBlock(
                    in_channels=current_in_channels,
                    out_channels=current_out_channels,
                    kernel_size=3,
                    max_pool=i
                    < config.num_conv_layers - 1,  # last layer doesn't have max pool
                    maxpool_padding=maxpool_padding,
                )
            )
            current_in_channels = current_out_channels
            current_out_channels = current_out_channels * 2
            if i < config.num_conv_layers - 1:
                # Update width: (width + 2*padding) // 2
                final_image_width = (final_image_width + 2 * maxpool_padding) // 2
        self.conv_layers = nn.Sequential(*conv_layers)

        # Get the output channels of the last conv layer
        last_conv_out_channels = conv_layers[-1].conv.out_channels

        fc_layers = []
        fc_input_dims = [
            final_image_width * final_image_width * last_conv_out_channels
        ] + [config.fc_hidden_dim] * (config.num_fc_layers - 1)
        fc_output_dims = [config.fc_hidden_dim] * (config.num_fc_layers - 1) + [
            config.num_classes
        ]
        for i in range(config.num_fc_layers):
            layers = [
                nn.Dropout(p=config.dropout_rate),
                nn.Linear(fc_input_dims[i], fc_output_dims[i]),
            ]
            if i < config.num_fc_layers - 1:  # Don't activate after last layer
                layers.append(nn.SiLU(inplace=True))
            fc_layers.append(nn.Sequential(*layers))
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(
            x, start_dim=1
        )  # flatten all dimensions except batch dimension
        x = self.fc_layers(x)
        return x

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-entropy loss.

        Args:
            logits: Model output logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,) with integer class indices

        Returns:
            loss: Scalar tensor containing the cross-entropy loss
        """
        return self.loss_fn(logits, targets)
