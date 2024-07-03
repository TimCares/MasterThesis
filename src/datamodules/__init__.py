from .unimodal_datamodules import (
    BaseDataModule,
    IMDBDataModule,
    OpenWebTextDataModule,
    CIFARDataModule,
    ImageNetDataModule,
    LibriSpeechDataModule,
    SpeechCommandsDataModule,
    QQPDataModule,
    MRPCDataModule,
    UNIMODAL_DATAMODULE_REGISTRY
)

from .multimodal_datamodules import (
    COCOCaptionsDataModule,
    VisualGenomeDataModule,
    ConceptualCaptionsDataModule,
    VQAv2DataModule,
    NLVR2DataModule,
    Flickr30DataModule,
    Flickr8AudioDataModule,
    CommonVoiceDataModule,
    SBUCaptionsDataModule,
    MULTIMODAL_DATAMODULE_REGISTRY
)
from .kd_datamodules import (
    KDDataModule,
    KD_DATAMODULE_REGISTRY,
)

from .dummy import (
    DummyDataModule,
    DUMMY_DATAMODULE_REGISTRY
)

from .glue import (
    GLUE_DATAMODULE_REGISTRY
)

from .multi_data_module import MultiDataModule

DATAMODULE_REGISTRY = UNIMODAL_DATAMODULE_REGISTRY | MULTIMODAL_DATAMODULE_REGISTRY | KD_DATAMODULE_REGISTRY \
    | DUMMY_DATAMODULE_REGISTRY | GLUE_DATAMODULE_REGISTRY # combine them