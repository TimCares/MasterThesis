from .unimodal_datamodules import (
    BaseDataModule,
    IMDBDataModule,
    OpenWebTextDataModule,
    CIFARDataModule,
    ImageNetDataModule,
    LibriSpeechDataModule,
    SpeechCommandsDataModule,
    UNIMODAL_DATAMODULE_REGISTRY
)

from .multimodal_datamodules import (
    COCOCaptionsDataModule,
    VisualGenomeDataModule,
    VQAv2DataModule,
    NLVR2DataModule,
    Flickr30DataModule,
    Flickr8AudioDataModule,
    CommonVoiceDataModule,
    MULTIMODAL_DATAMODULE_REGISTRY
)
from .kd_datamodules import (
    KDDataModule,
    KD_DATAMODULE_REGISTRY,
)

DATAMODULE_REGISTRY = UNIMODAL_DATAMODULE_REGISTRY | MULTIMODAL_DATAMODULE_REGISTRY | KD_DATAMODULE_REGISTRY # combine them