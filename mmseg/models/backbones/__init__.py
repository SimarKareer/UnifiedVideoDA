# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional backbones

from .mix_transformer import (MixVisionTransformer, mit_b0, mit_b1, mit_b2,
                              mit_b3, mit_b4, mit_b5)
from .mix_transformer_linfus import (MixVisionTransformerLinearFusion, mit_b0_linfus, mit_b1_linfus,
                                      mit_b2_linfus, mit_b3_linfus, mit_b4_linfus, mit_b5_linfus)
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnext import ResNeXt

__all__ = [
    'ResNet',
    'ResNetV1c',
    'ResNetV1d',
    'ResNeXt',
    'ResNeSt',
    'MixVisionTransformer',
    'mit_b0',
    'mit_b1',
    'mit_b2',
    'mit_b3',
    'mit_b4',
    'mit_b5',
    'MixVisionTransformerLinearFusion',
    'mit_b0_linfus',
    'mit_b1_linfus',
    'mit_b2_linfus',
    'mit_b3_linfus',
    'mit_b4_linfus',
    'mit_b5_linfus',
]
