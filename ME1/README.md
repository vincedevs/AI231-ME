# ME1: Manual Three-Layer CNN for MNIST

This exercise trains a compact three-layer convolutional neural network on MNIST using low-level PyTorch tensor operations. See `mnist_manual_cnn.ipynb` for the complete implementation and executed results.

## Environment

From this directory, create the locked environment and launch the notebook with:

```bash
uv sync --locked
uv run jupyter notebook mnist_manual_cnn.ipynb
```

For a non-interactive, reproducible top-to-bottom execution:

```bash
CUDA_VISIBLE_DEVICES=0 uv run jupyter nbconvert \
  --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 \
  mnist_manual_cnn.ipynb
```

Dependencies are managed only by `uv`; the notebook contains no installation commands. MNIST is downloaded to the ignored `data/` directory.

## Architecture and manual operations

- Three same-padded 3×3 convolutions: `1 → 16 → 32 → 48` channels.
- ReLU uses the elementwise tensor operation `torch.clamp`.
- Two non-overlapping 2×2 downsampling stages use `einops.rearrange` and `einops.reduce(..., "max")`.
- The final `48 × 7 × 7` feature map is flattened with `einops.rearrange`.
- A manual 2,352-to-10 classifier produces digit logits.

Each convolution organizes sliding patches with tensor `unfold` and `einops.rearrange`, then contracts patches and `nn.Parameter` weights with `torch.einsum`. The classifier is another `torch.einsum` contraction over explicit `nn.Parameter` weights and biases. **Neither `nn.Conv2d` nor `nn.Linear` is used**, and no built-in activation or pooling module is used.

## Results

The notebook trains for exactly five epochs on the standard 60,000-image MNIST training split and evaluates on the official 10,000-image test split. The executed five-epoch results were:

| Epoch | Training loss | Training accuracy | Test accuracy |
|---:|---:|---:|---:|
| 1 | 0.2171 | 93.87% | 98.46% |
| 2 | 0.0455 | 98.58% | 98.74% |
| 3 | 0.0286 | 99.11% | 99.07% |
| 4 | 0.0173 | 99.45% | 99.09% |
| 5 | 0.0110 | 99.70% | **99.18%** |

The final accuracy was **99.18% (9,918/10,000)** on the official MNIST test split. The notebook also contains an executed 4×4 grid of 16 test predictions labeled with ground truth and prediction; correct titles are green and incorrect titles are red.

The notebook automatically chooses one visible CUDA device when available, otherwise CPU. A modest batch size of 256 and only 42,202 parameters conserve shared-server resources.
