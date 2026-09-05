# ConvergeFlow

Official implementation of **ConvergeFlow: Language Flow with Provable Convergence to Token Embeddings**.

ConvergeFlow is an embedding-space flow-based language model whose data predictor is constrained to the convex hull of the vocabulary embeddings. It is trained with the mean-squared-error objective induced by flow matching and can predict tokens directly, without a separately trained cross-entropy decoder. The paper proves convergence of the learned flow to valid token embeddings under suitable regularity conditions and introduces several inference-time controls for the quality-diversity trade-off.

## Model

The default configuration follows the paper's OpenWebText experiment:

| Component | Value |
|---|---|
| Transformer blocks | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| Context length | 1,024 |
| Vocabulary size | 50,257 |
| Parameters | approximately 130M |
| Self-conditioning probability | 0.25 |
| Training objective | weighted embedding-space MSE |

The implementation is initialized from the public LangFlow OpenWebText checkpoint. The vocabulary embedding matrix is frozen by default to prevent embedding collapse, while the predictor is fine-tuned with the ConvergeFlow objective.

## Installation

Python 3.10 or later and PyTorch with CUDA support are recommended. Multi-GPU training uses NCCL and therefore requires NVIDIA GPUs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy tqdm einops transformers
```

## Data preparation

Training expects a directory containing tokenized OpenWebText splits:

```text
openwebtext/
├── train.bin
└── val.bin
```

Each file must be a flat, headerless array of `uint16` token IDs. The default vocabulary size is 50,257, matching GPT-2. Training randomly reads contiguous windows of `block_size + 1` tokens from `train.bin`; the extra token is retained for compatibility with next-token-style packed datasets, although the current flow-matching loss uses the input window itself.

Any preprocessing pipeline may be used as long as it produces this format. For example:

```python
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
ids = tokenizer("your training text", add_special_tokens=False)["input_ids"]
np.asarray(ids, dtype=np.uint16).tofile("openwebtext/train.bin")
```

For a full experiment, concatenate and tokenize the complete training and validation corpora rather than a single document.

## Training

Training is launched with `torchrun`, including for a single GPU because `train.py` initializes a distributed process group.

Four GPUs:

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --data_path /path/to/openwebtext \
  --config_path config.json \
  --model Continuous-Rivals-Discrete/langflow-owt \
  --results_dir results \
  --batch_size 480 \
  --grad_accu 6 \
  --lr 1e-5 \
  --ckpt_every 25000
```

`batch_size` is the global batch size and must be divisible by `world_size * grad_accu`. The paper uses a global batch size of 480, AdamW with learning rate `1e-5`, and four or eight NVIDIA A100 40 GB GPUs.

Each run creates a timestamped directory:

```text
results/000-YYYYMMDD-HHMMSS/
├── log.txt
└── checkpoints/
    ├── 0025000.pt
    └── ...
```

By default, token embeddings remain frozen. Pass `--embed_trainable` only for ablations; jointly learning embeddings with the MSE objective can lead to embedding collapse.

## Sampling and evaluation

`evaluation.py` generates fixed-length unconditional samples, decodes them with the GPT-2 tokenizer, and reports:

- **Generative perplexity**, scored by GPT-2 Large by default.
- **Negative log-likelihood**, averaged over valid scorer tokens.
- **Unigram entropy**, averaged over generated token sequences.

Example:

```bash
python evaluation.py \
  --ckpt_path /path/0175000.pt \
  --config_path config.json \
  --num_samples 1024 \
  --batch_size 16 \
  --seq_length 1024 \
  --NFE 64 \
  --time_schedule ConvergeFlow \
  --config_type 1 \
  --wk 1 \
  --wscg 1.0 \
  --wug 0.0
```

To load the Hugging Face initialization instead of a local checkpoint, pass `--ckpt_path None`.

### Sampling controls

| Argument | Meaning | Default |
|---|---|---:|
| `--NFE` | Total neural-function-evaluation budget | 256 |
| `--time_schedule` | `ConvergeFlow` midpoint grid or the original `LangFlow` grid | `ConvergeFlow` |
| `--config_type` | Configuration A/B/C; see below | 1 |
| `--wk` | Nominal iterative self-conditioning depth | 1 |
| `--wscg` | Self-conditioning guidance strength | 1.0 |
| `--wug` | Unconditional guidance strength | 0.0 |
| `--pred_form` | Final token rule: weight argmax or nearest embedding | `weight` |
| `--tf32` / `--no-tf32` | Enable or disable TF32 CUDA kernels | enabled |

The configuration types implement the three allocation schedules studied in the paper:

| `config_type` | Refinement/guidance allocation | Paper name |
|---:|---|---|
| 1 | Constant across time | Configuration A |
| 2 | Scaled by `1 / (1 + sqrt(sigma / alpha))` | Configuration B |
| 3 | Scaled by `1 / (1 + sigma / alpha)` | Configuration C |

The standard one-step self-conditioned sampler corresponds to `wk=1`, `wscg=1`, and `wug=0`. Increasing guidance or refinement generally lowers perplexity at the cost of diversity. Under a fixed NFE budget, the sampler automatically reduces the number of solver steps to account for extra self-conditioning evaluations.
