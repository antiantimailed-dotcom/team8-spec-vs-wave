from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torchaudio
from torch.utils.data import Dataset

GTZAN_GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


class GTZANDataset(Dataset):
    """GTZAN genre clips as either mel-spectrograms or raw waveforms."""

    def __init__(
        self,
        root: str | Path,
        representation: Literal["spectrogram", "waveform"] = "spectrogram",
        sample_rate: int = 22050,
        duration_sec: float = 5.0,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        file_list: list[tuple[Path, int]] | None = None,
    ) -> None:
        self.root = Path(root)
        self.representation = representation
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration_sec)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        if file_list is not None:
            self.items = file_list
        else:
            self.items = []
            for label, genre in enumerate(GTZAN_GENRES):
                genre_dir = self.root / genre
                if not genre_dir.is_dir():
                    continue
                for path in sorted(genre_dir.glob("*.wav")):
                    if path.name.startswith("._"):
                        continue
                    self.items.append((path, label))

        if not self.items:
            raise FileNotFoundError(f"No GTZAN wav files found under {self.root}")

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self) -> int:
        return len(self.items)

    def _load_audio(self, path: Path) -> torch.Tensor:
        # Prefer soundfile: torchaudio>=2.11 routes load() through torchcodec.
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            wav = torch.from_numpy(data.T)  # (channels, samples)
        except Exception:
            wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.squeeze(0)
        if wav.numel() < self.num_samples:
            wav = torch.nn.functional.pad(wav, (0, self.num_samples - wav.numel()))
        else:
            # center crop for reproducibility
            start = max(0, (wav.numel() - self.num_samples) // 2)
            wav = wav[start : start + self.num_samples]
        return wav

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        wav = self._load_audio(path)
        if self.representation == "waveform":
            x = wav.unsqueeze(0)  # (1, T)
        else:
            spec = self.db(self.mel(wav))  # (n_mels, time)
            x = spec.unsqueeze(0)  # (1, n_mels, time)
        return x, label
