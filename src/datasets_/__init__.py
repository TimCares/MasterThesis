from .base_datasets import (
    BaseDataset,
    NLPDataset,
    AudioDataset,
    ImageDataset,
    BaseImageText,
    BaseImageAudio,
    BaseTextAudio,
)

from .unimodal_datasets import (
    OpenWebTextDataset,
    IMDBDataset,
    LibriSpeechDataset,
    SpeechCommandsDataset,
    ImageNetDataset,
    CIFARDataset,
    QQPDataset,
    UNIMODAL_DATASET_REGISTRY,
)

from .multimodal_datasets import (
    COCOCaptions, 
    Flickr30Dataset, 
    Flickr8KAudioDataset, 
    VisualGenome,
    VQAv2,
    NLVR2,
    CommonVoice,
    MULTIMODAL_DATASET_REGISTRY,
)

from .data_utils import (
    convert_mp3_to_flac,
    write_data_into_jsonl,
    download_and_unzip,
    get_transforms,
)

from .kd_datasets import (
    KDDataset,
    ImageKDDataset,
    AudioKDDataset,
    TextKDDataset,
    KD_DATASET_REGISTRY,
)

from .glue import (
    CoLA,
    SST,
    QNLI,
    RTE,
    MRPC,
    QQP,
    STSB,
    MNLI,
    GLUE_DATASET_REGISTRY,
)

DATASET_REGISTRY = UNIMODAL_DATASET_REGISTRY | MULTIMODAL_DATASET_REGISTRY | KD_DATASET_REGISTRY | GLUE_DATASET_REGISTRY