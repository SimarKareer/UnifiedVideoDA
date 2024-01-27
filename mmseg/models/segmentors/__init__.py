# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add HRDAEncoderDecoder

from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder
from .hrda_encoder_decoder import HRDAEncoderDecoder
from .accel_hrda_encoder_decoder import ACCELHRDAEncoderDecoder

__all__ = ['BaseSegmentor', 'EncoderDecoder', 'HRDAEncoderDecoder', 'ACCELHRDAEncoderDecoder']
