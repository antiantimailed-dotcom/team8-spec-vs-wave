# Significance tests

Accuracy on the test clips. Gap = spectrogram − SampleCNN (positive: spectrogram ahead). CIs and p-values: 10,000 bootstrap resamples of test clips, seeds averaged; McNemar: exact test per seed.

## RQ1 (FMA-small)

No full FMA-small runs found.

## RQ2: GTZAN random vs. fault-filtered split

| Comparison | Spectrogram | SampleCNN | Gap, pts [95% CI] | p (bootstrap) | Gap per seed | McNemar p per seed |
|---|---|---|---|---|---|---|
| GTZAN random | 83.8% | 73.8% | **+10.0** [+4.7, +15.6] | < 0.001 | +10.0, +5.3, +14.7 | 0.004, 0.169, 0.000 |
| GTZAN filtered | 62.1% | 60.7% | **+1.4** [-2.8, +5.5] | 0.521 | -2.1, -2.1, +8.3 | 0.539, 0.556, 0.001 |

**Drop difference:** the spectrogram CNN loses **+8.6 pts more** than SampleCNN when moving to the fault-filtered split (= gap on random − gap on filtered; 95% CI [+1.5, +15.6], p = 0.017).

