# vision data

Input datasets for the vision experiments.

## Expected subdirs

- `MNIST/` — MNIST digits (auto-downloaded by torchvision via `mnist_data.py`)
- `phys101/` — Phys101 video dataset, consumed by `phys101_dataset_explore.ipynb`
- `raw/` — other raw assets

## Download

MNIST downloads automatically on first use:

```
python mnist_data.py
```

Phys101 is not a standard package; request / download from:
https://phys101.csail.mit.edu/
