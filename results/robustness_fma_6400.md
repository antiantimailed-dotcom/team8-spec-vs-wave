# Robustness to degraded audio: FMA-small, 6,400 training clips

Test accuracy, mean ± SD over seeds. Drop = clean − degraded, paired by seed (larger = more fragile). 'Spectrogram drops more' counts seeds.

## FMA-small, 6,400 training clips

| Condition | Spectrogram | SampleCNN | Spectrogram drop | SampleCNN drop | Seeds where spectrogram drops more |
|---|---|---|---|---|---|
| clean | 54.5 ± 0.8 | 48.7 ± 0.4 | +0.0 | +0.0 | — |
| noise_snr20 | 49.2 ± 0.6 | 45.6 ± 2.2 | +5.2 | +3.1 | 2 of 3 |
| noise_snr10 | 36.3 ± 1.7 | 32.2 ± 0.5 | +18.2 | +16.5 | 2 of 3 |
| noise_snr0 | 29.2 ± 1.3 | 15.1 ± 1.7 | +25.3 | +33.6 | 0 of 3 |
| lowpass_4k | 43.7 ± 1.4 | 39.1 ± 2.0 | +10.8 | +9.6 | 2 of 3 |
| lowpass_2k | 37.9 ± 0.2 | 26.5 ± 1.0 | +16.6 | +22.2 | 0 of 3 |
| mp3_64k | 30.0 ± 3.0 | 48.6 ± 0.4 | +24.5 | +0.1 | 3 of 3 |
| mp3_32k | 23.1 ± 2.9 | 46.2 ± 0.5 | +31.4 | +2.5 | 3 of 3 |

