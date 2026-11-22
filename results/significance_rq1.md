# Significance tests

Accuracy on the test clips. Gap = spectrogram − SampleCNN (positive: spectrogram ahead). CIs and p-values: 10,000 bootstrap resamples of test clips, seeds averaged; McNemar: exact test per seed.

## RQ1: gap by training size (FMA-small)

| Comparison | Spectrogram | SampleCNN | Gap, pts [95% CI] | p (bootstrap) | Gap per seed | McNemar p per seed |
|---|---|---|---|---|---|---|
| FMA n=800 | 45.4% | 42.4% | **+3.0** [+0.3, +5.6] | 0.032 | +5.5, -1.7, +5.1 | 0.004, 0.379, 0.008 |
| FMA n=1,600 | 51.7% | 42.8% | **+9.0** [+6.3, +11.6] | < 0.001 | +11.9, +7.2, +7.7 | 0.000, 0.000, 0.000 |
| FMA n=3,200 | 49.8% | 46.4% | **+3.5** [+0.9, +6.1] | 0.011 | +6.0, +4.4, +0.0 | 0.002, 0.017, 1.000 |
| FMA n=6,400 | 54.5% | 48.7% | **+5.8** [+3.5, +8.1] | < 0.001 | +4.9, +6.9, +5.6 | 0.006, 0.000, 0.001 |

**Trend:** the gap changes by **+0.30 pts per doubling** of the training set (95% CI [-0.76, +1.35], p = 0.592, 800 shared test clips): no evidence the gap changes with training size.

