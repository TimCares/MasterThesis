from .multimodal_datamodules import *
from .unimodal_datamodules import *

REGISTRY = UNIMODAL_REGISTRY | MULTIMODAL_REGISTRY # combine them