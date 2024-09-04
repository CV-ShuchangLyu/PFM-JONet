# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmseg.registry import MODELS

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS

## added by LYU: 2024/07/24
DISCRIMINATORS = MODELS

## added by LYU: 2024/07/24
def build_discriminator(cfg):
    """Build a module or modules from a list."""
    #return build_module(cfg, DISCRIMINATORS, default_args)
    return DISCRIMINATORS.build(cfg)


def build_backbone(cfg):
    """Build backbone."""
    warnings.warn('``build_backbone`` would be deprecated soon, please use '
                  '``mmseg.registry.MODELS.build()`` ')
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    warnings.warn('``build_neck`` would be deprecated soon, please use '
                  '``mmseg.registry.MODELS.build()`` ')
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    warnings.warn('``build_head`` would be deprecated soon, please use '
                  '``mmseg.registry.MODELS.build()`` ')
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    warnings.warn('``build_loss`` would be deprecated soon, please use '
                  '``mmseg.registry.MODELS.build()`` ')
    return LOSSES.build(cfg)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
'''
## added by LYU: 2022/03/29
def build_module(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.
    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.ModuleList(modules)

    return build_from_cfg(cfg, registry, default_args)
'''