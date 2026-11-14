# Start here

This folder is a complete, runnable Advanced Computer Vision assignment project.

## What the experiment does

Each RGB image is a grid of colour values. Its manual mask marks spacecraft pixels white and background pixels black. The network learns to predict this mask. The encoder compresses the image into feature grids. The decoder enlarges those features into a pixel classification. Optional skips add encoder features directly to the decoder to test whether access to detail helps.

## Windows setup from an empty folder

Install Python 3.11 or 3.12 from python.org if needed. Extract this project ZIP. Open PowerShell in the extracted `satseg_project` folder. These commands use the virtual environment Python directly, so no activation-policy change is needed:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe sanity_check.py
.\.venv\Scripts\python.exe train.py --epochs 40 --seed 42 --threads 2
.\.venv\Scripts\python.exe evaluate.py --threads 2
.\.venv\Scripts\python.exe make_report.py
```

Use `py -3.11` if that is your installed version. On macOS/Linux substitute `python3 -m venv .venv`, then `.venv/bin/python` for the Windows Python path. If your machine has an available CUDA GPU and compatible PyTorch, the scripts select it automatically; otherwise they run on CPU.

The delivered folder already includes a measured run. You can inspect its report and plots immediately; retraining is optional for reviewing the included evidence. Rerunning training overwrites the result files. Copy `results` first if you want to retain the original run.

## Folder contents

- `download_data.py`: obtains official SATSEG metadata and colour/mask files; checks byte counts and source MD5 checksums where supplied.
- `data.py`: verifies pairs, maps raw mask 0 to background and 255 to spacecraft, letterboxes inputs, saves explicit type-held-out splits and an audit.
- `model.py`: one configurable network; skip fusion is the only architectural difference.
- `train.py`: trains both models with matched settings and selects checkpoints by validation mIoU.
- `evaluate.py`: evaluates saved checkpoints on test images, times inference, creates all plots and `results/dashboard.html`.
- `sanity_check.py`: verifies correctness and learning on a tiny sample.
- `make_report.py`: creates the report from actual metrics and plots.
- `results`: measured logs, checkpoints, metrics, graphs, standalone dashboard and report.
- `data/train.txt`, `val.txt`, `test.txt`: exact scene IDs used in each partition.

## Open the visual results

Double-click `results/dashboard.html` in your browser. It contains the architecture, training curves, metrics, confusion matrices, and automatically selected failure/success examples. It runs offline.

## Fairness rules

Do not select epochs using test scores, change only one model's training budget, mix the grayscale duplicate of a scene into another split, or describe a negative result as an improvement. Both models use the same seed and sample order. The validation set contains Radarsat scenes only; the test set holds out other spacecraft types. This is a strict but small-data experimental protocol, not the official SATSEG benchmark split.
