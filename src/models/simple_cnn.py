import pydantic
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNConfig(pydantic.BaseModel):
    """
    Configuration for the SimpleCNN model, which performs image classification.
    """
    num_classes: int = 10 # output classes
    num_channels: int = 3
    num_filters: int = 16
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1

class SimpleCNN(nn.Module):
    pass