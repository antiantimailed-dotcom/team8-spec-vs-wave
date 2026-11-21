# Robustness to degraded audio

Test accuracy, mean ± SD over seeds. Drop = clean − degraded, paired by seed (larger = more fragile). 'Spectrogram drops more' counts seeds.

## GTZAN fault-filtered

| Condition | Spectrogram | SampleCNN | Spectrogram drop | SampleCNN drop | Seeds where spectrogram drops more |
|---|---|---|---|---|---|
| clean | 62.1 ± 3.3 | 60.7 ± 4.7 | +0.0 | +0.0 | — |
| noise_snr20 | 39.1 ± 8.8 | 40.7 ± 5.3 | +23.0 | +20.0 | 1 of 3 |
| noise_snr10 | 28.6 ± 9.2 | 25.5 ± 1.2 | +33.4 | +35.2 | 2 of 3 |
| noise_snr0 | 22.1 ± 5.7 | 23.2 ± 1.3 | +40.0 | +37.5 | 2 of 3 |
| lowpass_4k | 50.1 ± 5.2 | 54.9 ± 1.7 | +12.0 | +5.7 | 3 of 3 |
| lowpass_2k | 46.2 ± 4.8 | 51.3 ± 1.7 | +15.9 | +9.4 | 2 of 3 |
| mp3_64k | 61.8 ± 0.5 | 61.7 ± 4.0 | +0.2 | -1.0 | 2 of 3 |
| mp3_32k | 55.4 ± 0.8 | 57.0 ± 2.0 | +6.7 | +3.7 | 2 of 3 |

