From: https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec

$ python wav2vec_manifest.py /path/to/waves --dest /manifest/path --ext $ext --valid-percent $valid
=>
$ python wav2vec_manifest.py ../../../data/librispeech_finetuning/1h --dest ../../../data/wer-scoring --ext flac --valid-percent 0

Path to librispeech_finetuning is for 1h subset, necessary path was derived from:
https://pytorch.org/audio/stable/_modules/torchaudio/datasets/librilight_limited.html
-> Variable "_SUBSET_MAP"

split=train
$ python libri_labels.py /path/to/tsv --output-dir /output/dir --output-name $split
=>
$ python libri_labels.py ../../../data/wer-scoring/train.tsv --output-dir ../../../data/wer-scoring --output-name train

## For WER-Scoring on test-other of Librispeech (Data2Vec Table 2)

$ python wav2vec_manifest.py ../../../data/LibriSpeech/test-other --dest ../../../data/wer-scoring --ext flac --valid-percent 0

$ mv ../../../data/wer-scoring/train.tsv ../../../data/wer-scoring/test_other.tsv

$ python libri_labels.py ../../../data/wer-scoring/test_other.tsv --output-dir ../../../data/wer-scoring --output-name test_other