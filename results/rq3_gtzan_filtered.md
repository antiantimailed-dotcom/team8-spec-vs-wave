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

## SampleCNN filter frequencies (pure-tone probe)

Each channel's preferred frequency = the test tone it responds to most. Only **tuned** channels count (best tone ≥ 2× the channel's median response). Control: the same probe on untrained SampleCNNs with random weights.

Reference shares below 1 kHz: mel scale (128 bands) 30%, linear scale 9%. A channel with no real preference lands on a random test tone, and 60% of the tones are below 1 kHz.

| Layer | Trained: tuned channels | Trained: median preferred | Trained: below 1 kHz | Untrained: tuned channels | Untrained: median preferred | Untrained: below 1 kHz |
|---|---|---|---|---|---|---|
| stem | 175 of 384 (46%) | 10,000 Hz | 0% | 19 of 384 (5%) | 10,000 Hz | 0% |
| block 1 | 245 of 384 (64%) | 8,640 Hz | 0% | 95 of 384 (25%) | 4,316 Hz | 9% |
| block 2 | 204 of 384 (53%) | 6,940 Hz | 4% | 65 of 384 (17%) | 5,181 Hz | 35% |
| block 3 | 583 of 768 (76%) | 5,374 Hz | 41% | 100 of 768 (13%) | 2,323 Hz | 42% |

Read it this way: a learned, mel-like frequency layout shows up as **many more tuned channels than the untrained control**, with a share below 1 kHz near the mel reference rather than the random-tone level.

