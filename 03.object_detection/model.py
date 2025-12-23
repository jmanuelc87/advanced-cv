import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv
import torchvision.tv_tensors as tvt
import torchvision.models.feature_extraction as fe

import ultralytics as ult
import ultralytics.nn.modules as ult_nn


class Backbone(nn.Module):
    """Backbone for my custom object detection algorithm
    currently suports two models:

        1. EfficientNet V2 m
        2. ConvNeXt Base

    Args:
        nn (nn.Module): the base module to extract features from intermediate layers
    """

    def __init__(
        self,
        from_model: nn.Module,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.layers, in_channels, out_channels = self.get_layers(
            from_model.__class__.__name__
        )

        self.backbone = fe.create_feature_extractor(
            from_model, return_nodes=self.layers
        )

        self.c3k20 = ult_nn.C3k2(
            c1=in_channels[0], c2=out_channels[0], n=2, c3k=False, e=0.25
        )
        self.c3k21 = ult_nn.C3k2(c1=in_channels[1], c2=out_channels[1], n=2, c3k=True)
        self.c3k22 = ult_nn.C3k2(c1=in_channels[2], c2=out_channels[2], n=2, c3k=True)
        self.sppf = ult_nn.SPPF(out_channels[2], 1024)
        self.c2spa = ult_nn.C2PSA(out_channels[2], 1024)

    def get_layers(
        self, model_name: str
    ) -> tuple[dict[str, str], list[int], list[int]]:
        """Retrives the layer names, input channels and output channels
        given a model.

        Args:
            model_name (str): The model name

        Returns:
            tuple[dict[str, str], list[int], list[int]]: dict of layers to extract the features, in channels and out channels
        """
        l = []
        in_ch = []
        out_ch = [512, 512, 1024]

        if model_name == "ConvNeXt":
            l = range(2, 8, 2)
            in_ch = [256, 512, 1024]
        elif model_name == "EfficientNet":
            l = range(5, 8, 1)
            in_ch = [176, 304, 512]

        return {f"features.{k}": f"layer{i}" for i, k in enumerate(l)}, in_ch, out_ch

    def forward(self, x: torch.Tensor):
        """Performs a forward pass on the input tensor

        Args:
            x (torch.Tensor): the input tensor

        Returns:
            torch.Tensor: the output p1, p2 and p3 of the base model
        """
        feat = self.backbone(x)
        p3 = self.c3k20(feat["layer0"])
        p2 = self.c3k21(feat["layer1"])
        p1 = self.c3k22(feat["layer2"])
        p1 = self.sppf(p1)
        p1 = self.c2spa(p1)

        return p1, p2, p3
