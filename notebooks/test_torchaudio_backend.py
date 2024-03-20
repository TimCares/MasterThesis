import torchaudio

# Replace the path with the path to your actual FLAC file
file_path = '../notebooks/data/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac'

waveform, sample_rate = torchaudio.load(file_path)
print(f"Waveform shape: {waveform.shape}, Sample rate: {sample_rate}")