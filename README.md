# WRTA-Net

Implementation of the Whitened-Residual Attention Network (WRTA-Net) for replay-attack localisation and typing in PMU-assisted power system state estimation.

For each 12-s window of RTU meters, the network outputs a replayed-meter score per meter and a four-class attack type (Normal, MDDA, LT-RDCA, ST-RDCA). The parameters are shared across meters, so the network size (71,398 parameters) does not depend on the number of meters or on the grid.

## Architecture

- **Input** per meter and scan, shape `(batch, 12, m, 5)`: whitened PMU-only residual, clone feature, PTDF vulnerability bit, log robust noise scale, topology flag.
- **Meter-shared state-space encoder**: S4D convolution layers with FiLM conditioning on the log noise scale.
- **Physics-biased multi-head attention**: bias `beta * A_ij` toward meters that share a bus; vulnerability-bit dropout (p = 0.3) during training.
- **Heads**: meter logit (replayed or not) and four-class logit.

## Files

| File | Content |
|---|---|
| `src/nets.py` | `WRNet` (proposed network) and the neural benchmark architectures |
| `src/train_dl.py` | model construction, weighted cross-entropy plus weighted BCE training loop, validation-based checkpoint and threshold selection |
| `src/mldata.py` | scenario pool container, clone feature, robust scale and evaluation metrics |

## Usage

```python
import numpy as np, torch
from src.nets import WRNet

m, W = 55, 12                                   # number of meters, window length
kinds = np.random.randint(0, 5, m)              # meter kind: V, P, Q, P_f, Q_f
vbit = np.random.randint(0, 2, m)               # PTDF vulnerability bit
A = (np.random.rand(m, m) < 0.1).astype(np.float32)
A = np.maximum(A, A.T); np.fill_diagonal(A, 1)  # bus-sharing adjacency of the meters

net = WRNet(m, W, A, kinds, vbit, d=32, temporal="ssm", graph=False, film=True,
            layers=3, glayers=1, attn=True, heads=4, lag=True, attn_bias=True,
            prior=True, prior_drop=0.3)
x = torch.randn(8, W, m, 5)                     # (batch, window, meters, channels)
class_logits, meter_logits = net(x)             # (8, 4), (8, m)
```

`net.set_system(A, kinds, vbit)` swaps the measurement graph, which allows applying a trained network to another grid.

## Training configuration

AdamW (learning rate 1.10e-3, weight decay 1.75e-3), one-cycle schedule (15 % warm-up), batch size 64, 20 epochs, gradient clipping 2.0, hidden size d = 32, 3 temporal layers, 1 attention layer, 4 heads, loss = class-weighted cross-entropy + 4.0 x weighted BCE on the meter mask. The meter threshold is chosen on validation data by maximising the attacked-meter F1.

## Requirements

Python 3.13, PyTorch, NumPy, scikit-learn.
