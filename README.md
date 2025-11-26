Here is the updated `README.md` tailored to your new 90-minute, 3.25 loss run using FineWeb-Edu.

````markdown
# NanoGPT-124M — In a Cave With a Box of Scraps

This is an effort to speedrun training NanoGPT (GPT-2 124M) on a **single consumer RTX 4090** from scratch using **FineWeb-Edu** data. The goal was to hit < 3.28 validation loss as fast as possible.

**We achieved 3.25 loss in just 90 minutes.**

I have provided all training and inference code here, along with the trained model checkpoints at this Hugging Face link:
[https://huggingface.co/DevParker/NanoGPT-124m-In-A-Cave-With-A-Box-Of-Scraps](https://huggingface.co/DevParker/NanoGPT-124m-In-A-Cave-With-A-Box-Of-Scraps)

---

## Speedrun 🏁

**From-scratch NanoGPT / GPT-2 124M training** on single-GPU consumer hardware.

**THE GOAL:**
1. **Achieve validation loss ≤ 3.28** (OpenAI GPT-2 baseline) on a GPT-2 124M(-ish) model trained from scratch.
2. Use **single-GPU, consumer hardware** (RTX 4090).
3. Use high-quality open data (**FineWeb-Edu**).

### Current Record

| Rank | Trainer | Hardware | Tokens Trained | Val Loss | Time to Target | Throughput | Training Script |
|---|---|---|---|---|---|---|---|
| 🥇 1 | DevParker | 1× RTX 4090 | ~0.92B | **3.25** | ~90 minutes | ~160k tok/s (peak) | `train_gpt2_4090_90min_3_25loss.py` |

---

## RESULTS

Single-GPU, from-scratch GPT-2-style training to **3.25 validation loss** in about **90 minutes** on a **single RTX 4090**, with:

* **124M parameters** (compute equivalent)
* **1792 context length** (via progressive windowing)
* ~**0.92B tokens** trained
* Up to **~163,000 tokens/sec** effective training throughput
* **Architecture:** "Modded-NanoGPT" (Muon optimizer, FlexAttention, Value Embeddings, U-Net connections).

This repo is both a **research playground** and a **proof of concept**: you can train a highly capable GPT-2-class model *extremely fast* on home hardware using modern techniques.

---

## Key Ideas & Features

All of this is implemented in `train_gpt2_4090_90min_3_25loss.py`.

### Model Architecture
* **GPT-2-style, 124M scale:** 12 layers, 6 heads, 768-dim embedding.
* **Value Embeddings:** Three distinct `value_embeds` tables injected into attention as alternative value streams (adds capacity without FLOPs).
* **U-Net Skip Connections:** Long skip connections from encoder layers to decoder layers to improve gradient flow.
* **Rotary Embeddings (RoPE):** Replaces standard learned positional embeddings.
* **Tanh Logit Scaling:** Final logits are scaled `logits = 30 * tanh(logits / 30)` to stabilize FP8/BF16 training.

### Optimization & Efficiency
* **Muon Optimizer:** A Newton-Schulz-style optimizer for matrix parameters that orthogonalizes weights, providing massive convergence speedups.
* **FlexAttention:** Uses `torch.nn.attention.flex_attention` with custom block masks for document-causal masking.
* **Async Data Loading:** Custom `AsyncDataLoader` with pinned memory to saturate the GPU (zero PCIe bottlenecks).
* **Curriculum Learning:** Progressive context window (starts at 256, ends at 1792) to save compute in the early phase.
* **FP8 Training:** Enabled via `torchao` for linear layers.

---

## Installation

You need a bleeding-edge PyTorch environment to support `torch.compile`, `flex_attention`, and `torchao`.

```bash
conda create -n speedrun python=3.10 -y
conda activate speedrun

# Install PyTorch Nightly (Required for FlexAttention/TorchAO as of late 2024/early 2025)
pip install --pre torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/nightly/cu124](https://download.pytorch.org/whl/nightly/cu124)

# Install dependencies
pip install numpy tqdm datasets tiktoken torchao
````

-----

## Usage

### 1\. Prepare the Data

We use **FineWeb-Edu** (sample-10BT). This script downloads the data, tokenizes it with GPT-2 tokenizer, and saves it into binary shards.

```bash
python prepare_edu.py
```

  * This will create `data/finewebEDU/fineweb_edu_train_*.bin`.
  * It aims for \~3B tokens to ensure enough buffer for the run.

### 2\. Run the Speedrun (90 Minutes)

Launch the training script. It is pre-configured for a single 4090 (24GB VRAM).

```bash
python train_gpt2_4090_90min_3_25loss.py
```

**What to expect:**

  * **Steps 0-300:** Rapid loss drop (10.0 -\> 4.5). High throughput (\~160k tok/s).
  * **Step \~300 & \~600:** "Lag Spikes" as the context window doubles (recompilation).
  * **Step 1100+:** Cooldown phase begins. Loss will plummet from \~3.3 to **3.25**.
  * **Step 1750:** Training finishes.

### 3\. Inference

Load your checkpoint and generate text.

```python
import torch
from train_gpt2_4090_90min_3_25loss import GPT, GPTConfig

device = "cuda"
# Update path to your actual checkpoint
ckpt_path = "gpt2_speedrun_FINAL.pt" 

ckpt = torch.load(ckpt_path, map_location=device)
model = GPT(GPTConfig()).to(device).bfloat16()
model.load_state_dict(ckpt)
model.eval()

# ... (See inference_standalone.py for full generation code)
```

-----

## Acknowledgements & Inspiration

  * **NanoGPT** by Andrej Karpathy for the baseline.
  * **Keller Jordan & the Modded-NanoGPT Community** ([GitHub](https://github.com/KellerJordan/modded-nanogpt)) for the 8xH100 speedrun concepts (Muon, Value Embeddings) that I adapted for single-GPU.
  * **HuggingFaceFW** for the incredible **FineWeb-Edu** dataset.
  * The **PyTorch Team** for `torch.compile` and `FlexAttention`.

-----

license: mit

```
