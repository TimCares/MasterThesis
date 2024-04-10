from .multimodal_datamodules import *
from .unimodal_datamodules import *
from .kd_datamodules import *

REGISTRY = UNIMODAL_REGISTRY | MULTIMODAL_REGISTRY | KD_DATAMODULE_REGISTRY # combine them