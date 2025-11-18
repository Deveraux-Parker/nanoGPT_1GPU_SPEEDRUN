# NanoGPT-124M — In a Cave With a Box of Scraps

This is an effort to speedrun training NanoGPT (gpt-2 124M) on a single consumer 4090 from scratch using FineWeb data. The goal is to hit 3.29 validation loss (or less) as fast as possible using just one single consumer 4090.

I will set a baseline and provide all training an inference code here, along with the trained model checkpoints at this huggingface link:
https://huggingface.co/DevParker/NanoGPT-124m-In-A-Cave-With-A-Box-Of-Scraps

We'll be using data from a cached version of FineWeb10B that is already tokenized with the GPT-2 tokenizer to save us some preprocessing.
hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=fname,
                  repo_type="dataset", local_dir=local_dir)

---

## Speedrun Leaderboard 🏁

**From-scratch NanoGPT / GPT-2 124M training** *single*-GPU, consumer hardware, while still reaching a solid language-modeling loss.

**Rules (for leaderboard submissions):**

1. **Achieve validation loss ≤ 3.29** on a GPT-2 124M(-ish) model trained from scratch on **FineWeb** using a NanoGPT-style setup.
2. Use **single-GPU, consumer hardware** (e.g., 4090 / 3090 / 4080, etc.).
3. Report:
   - Hardware (GPU model, VRAM)
   - Total wall-clock training time (start of step 0 to final validation)
   - Effective tokens seen
   - Final validation loss and step
   - Training script + exact command used to run it

> PRs welcome! Add a row to the table below and link to your log / run config.

### Current Record

| Rank | Trainer    | Hardware     | Tokens Trained | Val Loss | Time to Target | Throughput (approx) | Training Script        | Command                            |
|------|-----------|--------------|----------------|----------|----------------|---------------------|------------------------|-------------------------------------|
| 🥇 1 | DevParker | 1× RTX 4090  | ~0.92B         | **3.286** @ step 1750 | ~115 minutes | ~130–140k tokens/s      | `train_gpt_improved.py` | `python train_gpt_improved.py`     |

If you beat this, open a PR updating the table with your numbers and a short description of what you changed (hyperparams, architecture tweaks, etc.).

---

## RESULTS

Single-GPU, from-scratch GPT-2-style training to **3.286 validation loss** in about **115 minutes** on a **single RTX 4090**, with:

* **124M parameters**
* **1024 context length**
* ~**0.92B tokens** trained
* Up to **~130–140k tokens/sec** effective training throughput
* Lots of helpful custom architecture (U-Net-ish GPT-2, Muon optimizer, FlexAttention, smear/backout tricks) 

This repo is both a **research playground** and a **proof of concept**: you can train a reasonably capable GPT-2-class model *fast* on consumer hardware.

---

## Key Ideas & Features

All of this is implemented in `train_gpt_improved.py`. 

### Model

* **GPT-2-style, 124M scale**

  * 12 layers, 6 heads, 768-dim embedding
  * Vocabulary padded to 50304 (multiple of 128)

* **context length**

  * `sequence_length = 32 * 1024`
  * Rotary position embeddings on Q/K

* **U-Net-style encoder/decoder**

  * 6 encoder layers + 6 decoder layers
  * Learnable `skip_weights` to blend encoder states into decoder
  * “Backout” mechanism: store a mid-layer representation (`x_backout`) and subtract it at the end, scaled by a learned `backout_lambda`

* **Custom value pathways**

  * Three distinct `value_embeds` (nn.Embedding) injected into attention as alternative value streams
  * A learnable mixing scalar `lamb` within attention to interpolate between standard V and the value embeddings

* **Smear gate (token mixing)**

  * Learns to mix each token with the previous one:

    * Compute a “smear gate” on token features
    * For positions `t > 0`, add a gated fraction of token `t-1` into token `t`, scaled by `smear_lambda`
  * This is applied right after word embeddings and before the transformer stack

* **Gated attention heads**

  * Each layer has an `attn_gate` that produces per-head gates from the input, controlling how much each head contributes at each position

* **Tanh logit scaling**

  * Final logits are passed through:
    `logits = 30 * tanh(logits / 30)`
  * Used consistently in both training and inference to tame extreme logits and stabilize high LR training

---

### Attention & Context Tricks

* **FlexAttention with block masks**

  * Uses `torch.nn.attention.flex_attention` with a custom `block_mask`:

    * **Causal**: no looking forward
    * **Document-local**: no attention across document boundaries (based on `50256` as a separator)
    * **Windowed**: only attend within a sliding window of size `attn_blocksize`

* **Progressive attention window (curriculum)**

  * `attn_blocksize` grows over training:

    * Starts at a small window (e.g. ~64)
    * Increases toward **1792 tokens**
  * Early training uses short-range attention (cheaper, more stable), later training unlocks longer context

---

### Optimizer Stack (a bit wild)

Four separate optimizer groups:

1. **Embeddings & value embeddings (`optimizer1`)**

   * Adam, **LR = 0.6**
   * Includes token embeddings and the 3 value embedding tables

2. **LM head (`optimizer2`)**

   * Adam, **LR = 0.008**
   * Just the output projection

3. **Matrix parameters (`optimizer3`)**

   * Custom **Muon** optimizer
   * Applies a Newton–Schulz–style orthogonalization (`zeropower_via_newtonschulz5`) to gradients of 2D weight matrices
   * `lr = 0.05`, momentum 0.95, Nesterov style
   * Momentum is **warmed up** from 0.85 → 0.95 over the first 300 steps

4. **Scalar / small parameters (`optimizer4`)**

   * Adam, **LR = 0.04**
   * Includes things like:

     * Layer `lambdas` (per block mixing scalars)
     * `skip_weights` (U-Net skips)
     * `smear_lambda`, `backout_lambda`

* **LR schedule**

  * No warmup (`warmup_iters = 0`)
  * Flat LR until the last 640 steps
  * Linear cooldown during the last `cooldown_iters = 640` iterations (over 1750 total)

---

### Data Pipeline

* **Custom binary dataset format**

  * Shards named like `data/fineweb10B/fineweb_train_*.bin`
  * Header (256 × int32) includes:

    * Magic number (`20240520`)
    * Version (`1`)
    * Token count (`ntok`)
  * Body is `ntok` tokens as `uint16` (e.g. GPT-2 BPE IDs)

* **DistributedDataLoader (single GPU compatible)**

  * Streams through those shards one sequence at a time
  * Each training batch:

    * Sequence length `T = 32768`
    * Uses `x = tokens[:-1]`, `y = tokens[1:]`
  * Moves through each shard in blocks of `T * num_processes` tokens and loops across files

* **Validation**

  * `val_tokens = 10,485,760` (10M tokens)
  * Validation uses `val_steps = val_tokens // T = 320` sequences per eval

---

## Training Setup & Results

* **Hyperparameters (core)**

  ```python
  batch_size       = 16          # gradient accumulation steps
  sequence_length  = 32 * 1024   # 32K context
  num_iterations   = 1750        # total optimizer steps
  val_loss_every   = 125
  val_tokens       = 10_485_760  # ~10M tokens of val data
````

* **Effective tokens per step**

  * 1 sequence per forward pass, length = 32,768
  * 16 gradient accumulation steps per optimizer step
  * → 32,768 × 16 = **524,288 tokens / optimizer step**

* **Total tokens trained**

  * 524,288 × 1750 ≈ **917,504,000 tokens** (~0.92B)

* **Final metrics (as reported in logs)**

  * `step:1750/1750 train_loss:3.1758`
  * `step:1750/1750 val_loss:3.2860`
  * Perplexity ≈ `exp(3.286) ≈ 26.7`

* **Throughput & runtime**

  * Wall-clock time ≈ **115 minutes** on a single RTX 4090
  * Effective training throughput ≈ **130k–140k tokens/sec**

---

## Repository Layout (typical)

* `train_gpt_improved.py`
  Main training script with:

  * Model definition (`GPT`, `Block`, `CausalSelfAttention`, etc.)
  * Data loader
  * Optimizers & schedulers
  * Training loop & logging

* `inference_standalone.py`
  Simple script to:

  * Load a saved checkpoint (`checkpoint_stepXXXXXX_lossY.YYYY.pt`)
  * Run a few canned prompts at different temperatures
  * Print generations to stdout

* `logs/`

  * Run logs and checkpoints:

    * `logs/<run_id>.txt`
    * `logs/<run_id>/checkpoint_step001750_loss3.2860.pt`

* `data/fineweb10B/` (not included, user-supplied)

  * Custom binary shards:

    * `fineweb_train_*.bin`
    * `fineweb_val_*.bin`

---

## Installation

You’ll need:

* A recent **PyTorch** build with:

  * `torch.compile`
  * `torch.nn.attention.flex_attention`
* **CUDA** + compatible driver
* **Triton** (installed automatically via recent PyTorch wheels)
* A GPU with at least **16–20 GB VRAM** (24 GB recommended for 32K context as configured here)

Example (conda):

```bash
conda create -n nanogpt-124m python=3.10 -y
conda activate nanogpt-124m

# Install PyTorch + CUDA (adjust command for your system)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Optional: other utilities
pip install numpy tqdm
```

> **Note:** The exact install line for PyTorch depends on your CUDA + OS; check the official PyTorch installation instructions if you run into issues.

---

## Data Preparation

This repo expects **pre-tokenized, binary** data compatible with the `DistributedDataLoader`:

* Each `*.bin` shard contains:

  * 256 × int32 header:

    * `header[0] = 20240520` (magic)
    * `header[1] = 1` (version)
    * `header[2] = ntok` (number of tokens)
  * `ntok` tokens as `uint16`

If you don’t already have data in that format, you’ll need a preprocessing script that:

1. Tokenizes your text (e.g., with GPT-2 tokenizer).
2. Writes the header & token buffer in the expected binary layout.

Paths are configured in `Hyperparameters`:

```python
@dataclass
class Hyperparameters:
    input_bin: str     = 'data/fineweb10B/fineweb_train_*.bin'
    input_val_bin: str = 'data/fineweb10B/fineweb_val_*.bin'
    ...
```

Update `input_bin` and `input_val_bin` to match your own dataset paths.

---

## Training

Once you have:

* Installed dependencies
* Prepared your dataset shards

You can launch training with:

```bash
python train_gpt_improved.py
```

This will:

* Initialize the 124M-parameter GPT model
* Start streaming training data from `input_bin`
* Periodically evaluate on `input_val_bin`
* Log to `logs/<run_id>.txt`
* Save checkpoints under `logs/<run_id>/checkpoint_stepXXXXXX_lossYYYY.pt`

Key knobs (edit in `Hyperparameters`):

* `batch_size` (gradient accumulation steps)
* `sequence_length` (context length, default 32K)
* `num_iterations`
* `val_loss_every`, `val_tokens`
* `cooldown_iters` (length of LR linear decay phase)

---

## Inference

The simplest way to play with the trained model is via `inference_standalone.py` or a small PyTorch snippet.

### Example (minimal PyTorch snippet)

```python
import torch
from train_gpt_improved import GPT, GPTConfig

device = "cuda"
ckpt_path = "logs/<run_id>/checkpoint_step001750_loss3.2860.pt"

ckpt = torch.load(ckpt_path, map_location=device)
model = GPT(GPTConfig()).to(device).bfloat16()
model.load_state_dict(ckpt["model"])
model.eval()

tokenizer = ...  # load GPT-2 tokenizer compatible with your training data

prompt = "The capital of France is"
input_ids = torch.tensor(tokenizer.encode(prompt), device=device, dtype=torch.long)[None]

with torch.no_grad():
    for _ in range(50):
        logits = model(input_ids[0], input_ids[0], attn_blocksize=torch.tensor(1792, device=device))
        logits = logits[:, -1, :]  # last-token logits
        next_id = torch.distributions.Categorical(logits=logits).sample()
        input_ids = torch.cat([input_ids, next_id[:, None]], dim=1)

print(tokenizer.decode(input_ids[0].tolist()))
```

You can also use the provided `inference_standalone.py` as a reference — it prints several test prompts with different temperatures and shows how the model behaves at the end of training.

---

## Acknowledgements & Inspiration

* **NanoGPT** by Andrej Karpathy for the “train GPT from scratch with minimal code” baseline.
* Modded NanoGPT speedruns and training code [https://github.com/KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (this is for 8xH100 but I adapted many of its features to this 1x4090 run).
* The PyTorch team for `torch.compile` and `flex_attention`, which make this kind of experiment actually feasible.
* Various community experiments with Muon, Triton kernels, and long-context training that inspired many of the tricks here.


---
license: mit
---
