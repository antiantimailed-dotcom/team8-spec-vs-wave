# RQ3: per-genre errors and learned filters

## GTZAN fault-filtered (3 spectrogram / 3 SampleCNN seeds)

| Genre | Spectrogram F1 | SampleCNN F1 | Difference | Seeds spectrogram better (of 3) |
|---|---|---|---|---|
| blues | 0.33 ± 0.10 | 0.11 ± 0.18 | +0.22 | 2 |
| metal | 0.93 ± 0.01 | 0.81 ± 0.04 | +0.12 | 3 |
| hiphop | 0.65 ± 0.13 | 0.61 ± 0.05 | +0.04 | 2 |
| classical | 0.96 ± 0.04 | 0.94 ± 0.01 | +0.03 | 2 |
| pop | 0.62 ± 0.06 | 0.63 ± 0.02 | -0.01 | 2 |
| reggae | 0.66 ± 0.06 | 0.69 ± 0.02 | -0.02 | 1 |
| jazz | 0.45 ± 0.12 | 0.48 ± 0.11 | -0.02 | 1 |
| disco | 0.60 ± 0.02 | 0.64 ± 0.01 | -0.03 | 0 |
| country | 0.69 ± 0.02 | 0.72 ± 0.03 | -0.04 | 0 |
| rock | 0.13 ± 0.05 | 0.31 ± 0.16 | -0.17 | 1 |

**Spectrogram CNN, most frequent confusions:** blues → jazz (45%); rock → disco (30%); hiphop → pop (23%); jazz → rock (23%); rock → pop (18%)
**SampleCNN, most frequent confusions:** blues → jazz (56%); hiphop → pop (33%); jazz → rock (28%); rock → pop (26%); blues → rock (24%)

## GTZAN random split (3 spectrogram / 3 SampleCNN seeds)

| Genre | Spectrogram F1 | SampleCNN F1 | Difference | Seeds spectrogram better (of 3) |
|---|---|---|---|---|
| hiphop | 0.82 ± 0.04 | 0.61 ± 0.08 | +0.21 | 3 |
| pop | 0.75 ± 0.05 | 0.54 ± 0.04 | +0.21 | 3 |
| blues | 0.98 ± 0.02 | 0.82 ± 0.09 | +0.16 | 3 |
| reggae | 0.77 ± 0.00 | 0.62 ± 0.00 | +0.15 | 3 |
| disco | 0.82 ± 0.03 | 0.72 ± 0.06 | +0.10 | 2 |
| country | 0.79 ± 0.01 | 0.72 ± 0.05 | +0.07 | 3 |
| metal | 0.93 ± 0.03 | 0.87 ± 0.03 | +0.06 | 3 |
| rock | 0.67 ± 0.07 | 0.63 ± 0.08 | +0.04 | 2 |
| classical | 0.97 ± 0.00 | 0.95 ± 0.06 | +0.02 | 1 |
| jazz | 0.83 ± 0.05 | 0.84 ± 0.12 | -0.01 | 2 |

**Spectrogram CNN, most frequent confusions:** pop → hiphop (18%); country → jazz (16%); pop → disco (11%); rock → jazz (11%); reggae → hiphop (9%)
**SampleCNN, most frequent confusions:** pop → hiphop (18%); disco → rock (16%); country → rock (13%); reggae → hiphop (13%); hiphop → reggae (13%)

