# Data

Point these paths at local copies of the datasets (do not commit audio).

Expected layout after setup:

```text
data/gtzan/genres/<genre>/*.wav
data/fma/fma_small/**/*.mp3
data/fma/fma_metadata/tracks.csv
```

On Ian's Mac Studio the helper script links:

- `~/Datasets/gtzan/genres` → `data/gtzan/genres`
- `~/Datasets/fma/fma_small` → `data/fma/fma_small`
- `~/Datasets/fma/fma_metadata` → `data/fma/fma_metadata`

Run:

```bash
./scripts/link_datasets.sh
```
