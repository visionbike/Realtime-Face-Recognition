import torch
import torch.nn.functional as fn
import torch.nn as nn


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1
) -> nn.Conv2d:
    """
    3x3 convolution with padding.

    :param in_planes: Number of input channels.
    :param out_planes: Number of output channels.
    :param stride: Stride of the convolution.
    :param groups: Number of blocked connections from input to output channels.
    :param dilation: Spacing between kernel elements (also used as padding).
    :return: 3x3 convolution layer.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    1x1 convolution with padding.

    :param in_planes: Number of input channels.
    :param out_planes: Number of output channels.
    :param stride: Convolution stride.
    :return: 1x1 convolution layer.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class IBasicBlock(nn.Module):
    """
    Improved ResNet basic block (BN-Conv-BN-PReLU-Conv-BN with residual add).
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1
    ):
        """
        :param in_planes: Number of input channels.
        :param out_planes: Number of output channels.
        :param stride: Stride of the second convolution.
        :param downsample: Optional module applied to the identity branch.
        :param groups: Must be 1 (only supported value).
        :param base_width: Must be 64 (only supported value).
        :param dilation: Must be 1 (dilation > 1 is not supported).
        """
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise ValueError("BasicBlock only supports dilation=1")
        self.bn1 = nn.BatchNorm2d(in_planes, eps=1e-5)
        self.conv1 = conv3x3(in_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes, eps=1e-5)
        self.prelu = nn.PReLU(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes, stride)
        self.bn3 = nn.BatchNorm2d(out_planes, eps=1e-5)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the block forward pass.

        :param x: Input feature map of shape (N, C, H, W).
        :return: Output feature map after residual add.
        """
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    """
    Improved ResNet backbone used by ArcFace.
    """

    fc_scale = 7 * 7    # spatial size (7x7) of the feature map before thr FC layer

    def __init__(
        self,
        block: type[IBasicBlock],
        layers: list[int],
        dropout: float = 0.0,
        num_features: int = 512,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: list[bool] | None = None,
        fp16: bool = False
    ):
        """
        :param block: Residual block type to stack.
        :param layers: Number of blocks per stage (4 elements).
        :param dropout: Dropout probability before the FC layer.
        :param num_features: Dimensionality of the output embedding.
        :param zero_init_residual: If true, zero-initialize the last BN in each block.
        :param groups: Number of convolution groups (must be 1).
        :param width_per_group: Base width per group (must be 64).
        :param replace_stride_with_dilation: Per-stage flags to use dilation instead of stride.
        :param fp16: If true, run the backbone under mixed-precision autocast.
        """
        super().__init__()
        self.fp16 = fp16
        self.in_planes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f"replace_stride_with_dilation should be None or a 3-element list, got {replace_stride_with_dilation}")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes, eps=1e-5)
        self.prelu = nn.PReLU(self.in_planes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features=num_features, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: type[IBasicBlock],
        out_planes: int,
        num_blocks: int,
        stride=1,
        dilate=False
    ) -> nn.Sequential:
        """
        Build one residual stage by stacking `blocks` instances of `block`.

        :param block: Residual block type to stack.
        :param out_planes: Number of output channels for the stage.
        :param num_blocks: Number of blocks in the stage.
        :param stride: Stride of the first block in the stage.
        :param dilate: If true, replace stride with dilation.
        :return: Sequential container holding the stage.
        """
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != out_planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, out_planes * block.expansion, stride),
                nn.BatchNorm2d(out_planes * block.expansion, eps=1e-5),
            )
        layers = [
            block(
                self.in_planes,
                out_planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
            )
        ]
        self.in_planes = out_planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    self.in_planes,
                    out_planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the L2-normalized face embedding for the batch of images.

        :param x: Input image batch of shape (N, 3, 112, 112).
        :return: L2-normalized face embedding of shape (N, num_features).
        """
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        x = fn.normalize(x, dim=1)
        return x


def _iresnet(
    block: type[IBasicBlock],
    layers: list[int],
    pretrained: bool = False,
    **kwargs
) -> IResNet:
    """
    Construct an IResNet model.

    :param block: Residual block type to stack.
    :param layers: Number of blocks per stage (4 elements).
    :param pretrained: Unsupported, weights are loaded via iresnet_inference instead.
    :param kwargs: Extra keyword arguments forwarded to IResNet.
    :return: Constructed IResNet model.
    """
    if pretrained:
        raise ValueError("pretrained weights are not supported; load a checkpoint via iresnet_inference().")
    return IResNet(block, layers, **kwargs)


def iresnet18(pretrained: bool = False, **kwargs) -> IResNet:
    """Build an 18-layer IResNet."""
    return _iresnet(IBasicBlock, [2, 2, 2, 2], pretrained, **kwargs)


def iresnet34(pretrained: bool = False, **kwargs) -> IResNet:
    """Build a 34-layer IResNet."""
    return _iresnet(IBasicBlock, [3, 4, 6, 3], pretrained, **kwargs)


def iresnet50(pretrained: bool = False, **kwargs) -> IResNet:
    """Build a 50-layer IResNet."""
    return _iresnet(IBasicBlock, [3, 4, 14, 3], pretrained, **kwargs)


def iresnet100(pretrained: bool = False, **kwargs) -> IResNet:
    """Build a 100-layer IResNet."""
    return _iresnet(IBasicBlock, [3, 13, 30, 3], pretrained, **kwargs)


def iresnet200(pretrained: bool = False, **kwargs) -> IResNet:
    """Build a 200-layer IResNet."""
    return _iresnet(IBasicBlock, [6, 26, 60, 6], pretrained, **kwargs)


def iresnet_inference(model_name: str, path: str, device: str | None = None) -> nn.Module:
    """
    Build an IResNet by name, load its weights, and return it in eval mode.

    :param model_name: Backbone name, one of {"r18", "r34", "r50", "r100"}.
    :param path: Path to the model weights (.pth state_dict).
    :param device: Target device. Defaults to "cuda" if available, otherwise "cpu".
    :return: Loaded model in evaluation mode.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "r18":
        model = iresnet18()
    elif model_name == "r34":
        model = iresnet34()
    elif model_name == "r50":
        model = iresnet50()
    elif model_name == "r100":
        model = iresnet100()
    else:
        raise ValueError(f"Unsupported model_name: {model_name!r}. Expected one of r18, r34, r50, r100.")

    weight = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(weight)
    model.to(device)

    return model.eval()
