from torchvision.models.detection.backbone_utils import BackboneWithFPN, LastLevelMaxPool
# from feature_pyramid_network import BackboneWithFPN
from typing import Callable, Dict, List, Optional, Union
import torchvision.models as tvmodels
# from torchvision.models import mobilenetv2, mobilenetv3,
import torch.nn as nn
# from .feature_pyramid_network import BackboneWithFPN, LastLevelMaxPool

"""
def resnet50_fpn_backbone(pretrain_path="",
                          norm_layer=nn.BatchNorm2d,
                          trainable_layers=3,
                          returned_layers=None,
                          extra_blocks=None):
"""
def mobilenet_fpn_backbone(backbone_model_name='mobilenet_v2',
                             trainable_layers:int = 0,
                             returned_layers: Optional[List[int]]=None,
                             extra_blocks=None,
                             norm_layer=nn.BatchNorm2d
                             ) ->nn.Module:
    if backbone_model_name == 'mobilenet_v2':
        mobilenet_model = tvmodels.mobilenet_v2(pretrained=True)
    elif backbone_model_name == 'mobilenet_v3':
        mobilenet_model = tvmodels.mobilenet_v3(pretrained=True)

    backbone =mobilenet_model.features
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    num_stages = len(stage_indices)

    # find the index of the layer from which we wont freeze
    if trainable_layers < 0 or trainable_layers > num_stages:
        raise ValueError(f"Trainable layers should be in the range [0,{num_stages}], got {trainable_layers} ")
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    out_channels = 256

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [num_stages - 2, num_stages - 1]

    if min(returned_layers) < 0 or max(returned_layers) >= num_stages:
        raise ValueError(f"Each returned layer should be in the range [0,{num_stages - 1}], got {returned_layers} ")

    return_layers = {f"{stage_indices[k]}": str(v) for v, k in enumerate(returned_layers)}
    # return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_list = [backbone[stage_indices[i]].out_channels for i in returned_layers]

    return BackboneWithFPN(backbone=backbone,
                           return_layers=return_layers,
                           in_channels_list=in_channels_list,
                           out_channels=out_channels,
                           extra_blocks=extra_blocks,
                           )


def convnext_fpn_backbone(backbone_model_name='convnext_tiny',
                             trainable_layers:int = 0,
                             returned_layers: Optional[List[int]]=None,
                             extra_blocks=None,
                             norm_layer=nn.BatchNorm2d
                             ) ->nn.Module:
    if backbone_model_name=='convnext_tiny':
        backbone_model = tvmodels.convnext_tiny(pretrained=True)
    elif backbone_model_name == 'convnext_small':
        backbone_model = tvmodels.convnext_small(pretrained=True)
    else:
        raise ValueError(f"backbone_model_name: choose from ['convnext_tiny','convnext_small']")
    # mobilenetv2_backbone= mobilenetv2(pretrained=True)
    backbone =backbone_model.features
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    num_stages = len(stage_indices)

    # find the index of the layer from which we wont freeze
    if trainable_layers < 0 or trainable_layers > num_stages:
        raise ValueError(f"Trainable layers should be in the range [0,{num_stages}], got {trainable_layers} ")
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    out_channels = 256

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [num_stages - 2, num_stages - 1]

    if min(returned_layers) < 0 or max(returned_layers) >= num_stages:
        raise ValueError(f"Each returned layer should be in the range [0,{num_stages - 1}], got {returned_layers} ")

    return_layers = {f"{stage_indices[k]}": str(v) for v, k in enumerate(returned_layers)}
    # return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_list = [backbone[stage_indices[i]].out_channels for i in returned_layers]

    return BackboneWithFPN(backbone=backbone,
                           return_layers=return_layers,
                           in_channels_list=in_channels_list,
                           out_channels=out_channels,
                           extra_blocks=extra_blocks,
                           )


