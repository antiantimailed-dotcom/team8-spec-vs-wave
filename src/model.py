"""Models. Every model takes a batch of raw 2.7 s slices, shape (B, 59049).

Main comparison (the proposal):
    spectrogram_cnn  log-mel 2D CNN, 5 conv blocks           about 1.11 M parameters
    sample_cnn       SampleCNN, Lee et al. 2017 (3^9 model)   about 1.59 M parameters
Ian's original small models, kept as extra baselines:
    small_spec_cnn   3 conv blocks on the same log-mel input  about 0.13 M parameters
    small_wave_cnn   3 strided 1D conv blocks                 about 0.24 M parameters
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio

from common import HOP, N_FFT, N_MELS, SAMPLE_RATE


class LogMel(nn.Module):
    """(B, L) waveform -> (B, 1, n_mels, frames) log-mel in dB, computed on the model's device."""
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            mel_scale="slaney", norm="slaney")          # matches librosa's defaults

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return (10.0 * torch.log10(self.mel(wav).clamp(min=1e-10))).unsqueeze(1)


# ---------------- main models ----------------
def _block2d(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2))


class SpectrogramCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.front = LogMel()
        self.norm = nn.BatchNorm2d(1)
        self.features = nn.Sequential(_block2d(1, 64), _block2d(64, 128), _block2d(128, 128),
                                      _block2d(128, 256), _block2d(256, 256))
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, n_classes))

    def forward(self, wav):
        x = self.features(self.norm(self.front(wav)))   # (B, 256, 4, 3)
        return self.head(x.amax(dim=(2, 3)))


def _block1d(cin, cout):
    return nn.Sequential(nn.Conv1d(cin, cout, 3, padding=1), nn.BatchNorm1d(cout),
                         nn.ReLU(inplace=True), nn.MaxPool1d(3))


class SampleCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 128, 3, stride=3), nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        ch = [128, 128, 128, 256, 256, 256, 256, 256, 256, 512]
        self.blocks = nn.Sequential(*[_block1d(ch[i], ch[i + 1]) for i in range(9)])
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, n_classes))

    def forward(self, wav):
        x = self.blocks(self.stem(wav.unsqueeze(1)))   # (B, 512, 1)
        return self.head(x.amax(dim=-1))


# ---------------- Ian's original models (unchanged layers) ----------------
class SmallSpecCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.front = LogMel()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)))      # (1,1) avoids MPS AdaptiveAvgPool2d limits
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 256),
                                        nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, wav):
        return self.classifier(self.features(self.front(wav)))


class SmallWaveCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=8, padding=28), nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=32, stride=4, padding=14), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=7), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 256),
                                        nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, wav):
        return self.classifier(self.features(wav.unsqueeze(1)))


MODELS = {
    "spectrogram_cnn": SpectrogramCNN,
    "sample_cnn": SampleCNN,
    "small_spec_cnn": SmallSpecCNN,
    "small_wave_cnn": SmallWaveCNN,
}


def build_model(name: str, n_classes: int) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown model {name}; choose from {list(MODELS)}")
    return MODELS[name](n_classes)
