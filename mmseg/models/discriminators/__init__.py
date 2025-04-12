# Copyright (c) OpenMMLab. All rights reserved.
from .AdapSeg import AdapSegDiscriminator
## added by LYU: 2025/04/10
from .wgangp import WGANGPDiscriminator

__all__ = [
    'AdapSegDiscriminator', 'WGANGPDiscriminator'
]
