# sun-dl-demo

Small PyTorch test run on the `sun` server: a 2-hidden-layer MLP (4,417 parameters)
that learns to separate two interleaved spirals (2,000 synthetic points, 80/20 train/test split).

## Files
- `demo.py` - CPU version
- `demo_gpu.py` - same model on one GPU (`CUDA_VISIBLE_DEVICES=0`)

## Setup (on the server)
```bash
python3 -m venv venv && ./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU
python3 -m venv venv-gpu && ./venv-gpu/bin/pip install torch                                           # CUDA
```

## Run
```bash
OMP_NUM_THREADS=4 ./venv/bin/python demo.py
CUDA_VISIBLE_DEVICES=0 ./venv-gpu/bin/python demo_gpu.py
```

## Results (run on sun, PyTorch 2.14.1)
| | CPU (4 threads) | GPU (RTX A5000) |
|---|---|---|
| Test accuracy | 99.0% | 99.0% |
| Training time (300 epochs) | ~2.4 s incl. startup | 0.52 s |
