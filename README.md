# LLM120: from-scratch English model and assistant on one 8 GB GPU

This repository trains a decoder-only language model **from random weights and a tokenizer trained
from scratch**. It is designed for this machine: an RTX 5050 Laptop GPU (8,151 MiB), an i7-13700HX,
24 GiB RAM, and more than 350 GiB free at project setup.

The model is intentionally close to the proven SmolLM2-135M design: 30 layers, width 576, a
1,536-wide SwiGLU MLP, 9 query heads, 3 KV heads (GQA), RMSNorm, RoPE, tied embeddings, and 2,048
token training sequences. A fresh 32,768-token byte-level BPE reduces the embedding table enough to
put the model near 125M parameters while keeping lossless fallback for text unseen by the English
tokenizer, including Devanagari. The same tokenizer contains a versioned chat template and explicit
system, user, and assistant tokens for later instruction tuning.

The complete pipeline has four distinct post-training stages: supervised instruction tuning (SFT),
offline preference tuning (DPO), a separately trained human-preference reward model, and a short,
KL-constrained online RLHF run using RLOO. DPO is not mislabeled as RLHF here.

## The important dataset decision

Do **not** begin by downloading 250 GB from the old SmolLM corpus. The complete old corpus is about
673 GB compressed (122 GB Cosmopedia v2 plus roughly 550 GB FineWeb-Edu-dedup) and its three
subsets contain about 252B tokens. A 250 GB prefix is not a principled mixture, and one laptop GPU
cannot realistically consume it once, much less reproduce the 600B-token SmolLM run.

The default here is [`EleutherAI/SmolLM2-135M-10B`](https://huggingface.co/datasets/EleutherAI/SmolLM2-135M-10B):
a 25.6 GB, 10B-token sample of the newer SmolLM2 mixture. It includes DCLM-Edu, FineWeb-Edu,
Cosmopedia v2, FineMath, Stack-Edu, and InfiMM-WebMath. This is the strongest practical first run
for the available compute and storage. It will occupy roughly 45–60 GB including raw Parquet,
uint16 tokens, and retained checkpoints.

This is still a serious run. Measure your actual throughput before committing: 10B tokens can take
weeks on a power-limited laptop GPU. The official SmolLM2-135M used 2T tokens on a cluster, so this
project aims for a sound, coherent small base model—not parity with the released SmolLM2 model.

## 1. Install

The RTX 5050 is an `sm_120` GPU. The project therefore uses PyTorch's CUDA 12.8 wheel index rather
than an older system CUDA build.

```bash
uv sync --extra dev --extra posttrain
source .venv/bin/activate
llm120-benchmark --vocab-size 32768 --steps 10
```

If `uv sync` reports that a newer PyTorch CUDA wheel has replaced cu128 for `sm_120`, update the
explicit index in `pyproject.toml`; do not install an old CUDA 12.1 wheel.

## 2. Run a small end-to-end pilot

An evenly spaced four-shard pilot catches installation, tokenizer, storage, and OOM problems before
the full download:

```bash
llm120-download --max-shards 4 --output data/raw/pilot
llm120-tokenizer \
  --input data/raw/pilot \
  --output artifacts/tokenizer-pilot \
  --max-documents 100000
llm120-prepare \
  --input data/raw/pilot \
  --tokenizer artifacts/tokenizer-pilot \
  --output data/tokenized-pilot \
  --workers 6
llm120-doctor --config configs/pilot.yaml
llm120-benchmark --config configs/pilot.yaml --steps 10
llm120-train --config configs/pilot.yaml --resume none
llm120-generate --output-dir checkpoints/llm120-pilot "The Earth revolves around"
```

The 20-step output is only a systems test and should still be nonsense. `ln(32768) = 10.40` is the
uniform-logit reference; with the official `1/sqrt(576)` initialization and tied embeddings, this
exact model measured about 10.9 on its first real batch, then should begin falling. A non-finite loss, a loss that never moves, or token
throughput collapsing to zero is a stop condition.

## 3. Prepare the production corpus

For the real run, start with a clean production raw-data directory and train the final tokenizer
from the full sample:

```bash
llm120-download --output data/raw/smollm2-135m-10b
llm120-tokenizer \
  --input data/raw/smollm2-135m-10b \
  --output artifacts/tokenizer-32k \
  --max-documents 500000
llm120-prepare \
  --input data/raw/smollm2-135m-10b \
  --tokenizer artifacts/tokenizer-32k \
  --output data/tokenized \
  --workers 16
llm120-doctor
```

Preprocessing is restartable. Completed input shards are fingerprinted and skipped; temporary files
from interrupted workers are rebuilt. Each document is enclosed by BOS/EOS boundaries. Validation documents are
chosen by a deterministic content hash, so they cannot also occur in training through a random row
split. The trainer reads memory maps directly and never loads the corpus into RAM.

Do not retrain or replace the tokenizer after tokenizing data or after starting model training. The
manifest and every checkpoint record SHA-256 identities and resume refuses mismatches.

## 4. Benchmark, then train

```bash
llm120-benchmark --steps 20
llm120-train
```

For the separate 400M-class run:

```bash
llm120-benchmark --config configs/train_400m.yaml --steps 20
llm120-train --config configs/train_400m.yaml --resume none
```

That run writes to `checkpoints/llm120-english-400m`, leaving the default
`checkpoints/llm120-english` run untouched. It has 404,827,200 parameters and trains on 10B tokens,
which is 24.7 tokens per parameter.

The default global batch is 65,536 tokens: four 2,048-token sequences in VRAM, accumulated for 8
microsteps. On this RTX 5050, repeated runs measured 6,460–7,449 tokens/s at 5.04 GiB peak allocated
VRAM (roughly 17–20 days for 10B tokens at 90% utilization, depending on power and thermals). It uses FP32 master weights and optimizer
state, BF16 compute, SDPA attention, activation
checkpointing, fused AdamW, gradient clipping, and a 3e-3 warmup-stable-linear-decay schedule. FP32
master state costs memory but is a deliberate stability choice for from-scratch training.

Checkpoints are written atomically every 1,000 optimizer steps; the last three are retained. `Ctrl-C`
or `SIGTERM` requests a checkpoint after the current optimizer update. Running `llm120-train` again
automatically resumes the newest checkpoint at the exact next data block, including optimizer,
scaler, and RNG state.

Monitor with:

```bash
tensorboard --logdir checkpoints/llm120-english/tensorboard
```

Keep the laptop plugged in, use its performance/cooling mode, prevent suspend, and watch thermals.
Do not enable `torch_compile` until the uncompiled benchmark and pilot are clean. Test it with
`llm120-benchmark --compile --steps 20`; retain it in the YAML only if it is faster and stable on the
installed PyTorch/Blackwell combination. It is disabled by default because the measured compiled
throughput on this machine was lower (5,864 tokens/s), despite reducing allocated VRAM to 3.51 GiB.

## 5. Evaluate the base model correctly

```bash
# Greedy continuation is the cleanest collapse/repetition diagnostic.
llm120-generate "Photosynthesis is the process by which"

# Sampling is useful only after greedy output is coherent.
llm120-generate --sample --temperature 0.8 --top-p 0.95 \
  "In a small village near the mountains,"
```

This is a **base completion model**, not a chat assistant. A prompt such as “Who are you?” is an
invalid capability test until supervised instruction tuning has been done. Test natural document
prefixes, held-out loss/perplexity, and downstream benchmarks. The inference command intentionally
does not hide repetition with `no_repeat_ngram_size` or a large repetition penalty.

Bad output after a nominal “100k steps” often means far fewer tokens than assumed. Here, a step is
explicitly 65,536 tokens, so 100,000 steps means 6.55B tokens. Other common causes addressed by this
pipeline are missing EOS boundaries, padding tokens included in loss, tokenizer/model vocabulary
mismatch, accidental checkpoint restart without optimizer state, unstable low-precision optimizer
state, excessive learning rate, bad sampling, and expecting instruction behavior from a base model.

Do not post-train a broken base model. Continue only when held-out loss is falling without spikes,
greedy continuations are recognizably English and locally coherent, and the same checkpoint produces
repeatable outputs. Instruction tuning can teach a response format; it cannot repair failed language
modeling or manufacture knowledge that the 125M model never learned.

## 6. Download and normalize all assistant data

The chosen datasets are deliberately small-model-oriented and fit comfortably on this machine:

- [`HuggingFaceTB/smol-smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk),
  the roughly 485K-example SFT set curated for SmolLM2-135M/360M;
- [`argilla/ultrafeedback-binarized-preferences-cleaned`](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned),
  roughly 61K preference pairs with known TruthfulQA contamination removed;
- [`nvidia/HelpSteer2`](https://huggingface.co/datasets/nvidia/HelpSteer2), human preference
  annotations used to train the independent reward model and supply online-RL prompts.

Download all three at immutable Hub revisions, then validate schemas and materialize deterministic
train/validation Parquet files:

```bash
llm120-download-posttrain
llm120-prepare-posttrain
```

The raw download manifest records every repository commit. Normalized data is written below
`data/posttrain/normalized`; malformed/empty conversations and preference ties are removed. RLHF
prompts are deduplicated and never contain the preferred answer.

## 7. Build the assistant

Run these only after the base-model quality gate above passes:

```bash
# 1. Learn instruction/Q&A/chat behavior; loss applies only to assistant turns.
llm120-sft

# 2. Learn chosen-over-rejected behavior from cleaned UltraFeedback pairs.
llm120-dpo

# 3. Train an independent scalar reward model from HelpSteer2 human comparisons.
llm120-reward

# 4. Run a conservative 250-step online RLHF phase from the DPO policy.
llm120-rlhf

# Test using the exact training chat template.
llm120-chat "What causes rain? Explain it to a ten-year-old."
```

Every trainer uses BF16 compute, TF32 matrix multiplication, FP32-loaded trainable weights, fused
AdamW, non-reentrant activation checkpointing, bounded sequence lengths, periodic validation, and
three rotating resumable checkpoints. Running a command again resumes automatically. Pass
`--max-steps 20 --resume none --output checkpoints/<new-smoke-directory>` for an isolated smoke run.

SFT uses a conservative 3e-4 cosine schedule and masks system/user tokens. DPO follows the official
small-SmolLM recipe's 1e-6 learning rate, 0.5 beta, and 1,024-token limit. The reward model uses
pairwise Bradley–Terry training plus reward centering. RLOO keeps a frozen reference policy, a KL
penalty, only two generations per prompt, short completions, and a small anti-degeneration auxiliary
reward. It refuses to start unless held-out reward-model preference accuracy is at least 55%.

Watch each stage rather than assuming more steps are better:

```bash
tensorboard --logdir checkpoints
```

For RLOO, stop if reward rises while KL, repetition, truncation, or qualitative answers worsen. Its
250-step default is an intentionally short first pass; inspect logged completions and checkpoints
before increasing it. Online RL is the easiest stage in which to reward-hack and degrade fluency.

A 125M assistant can become orderly and useful for simple questions, extraction, and basic
instructions, but it will remain capacity-limited in factual recall and multi-step reasoning. No
dataset or alignment algorithm can make it comparable to a modern multi-billion-parameter
assistant. The reliable objective is a coherent, measurable small model—not a promise that tuning
will erase the limits of its scale.

## 8. Later Nepali adaptation

The byte tokenizer can encode Nepali today, but inefficiently; that fallback is for correctness, not
quality. After the English model is stable, the recommended next phase is:

1. curate and deduplicate a high-quality Nepali corpus;
2. extend the tokenizer with Nepali merges while preserving every existing token ID;
3. initialize only the added embedding rows from their byte-token compositions;
4. continue pretraining with a Nepali-heavy mix plus 10–20% English replay;
5. evaluate Nepali fertility, held-out loss, and English forgetting before instruction tuning.

That vocabulary surgery should be a separate, checkpointed phase. Mixing it into this English run
would make the experiment harder to diagnose.

## Why these choices

- [SmolLM's training report](https://huggingface.co/blog/smollm) documents the original corpus,
  depth-first GQA architecture, tied embeddings, 2K context, 49,152-token tokenizer, trapezoidal
  schedule, and 600B-token training run.
- The [official SmolLM2 135M configuration](https://github.com/huggingface/smollm/blob/main/text/pretraining/smollm2/config_smollm2_135M.yaml)
  uses 30×576 layers, a 1,536 MLP, 9/3 GQA, BF16, AdamW β=(0.9, 0.95), gradient clipping at 1, and
  peak LR 3e-3. This pipeline adapts its global batch and memory strategy to one 8 GB GPU.
- The [SmolLM2 paper](https://arxiv.org/abs/2502.02737) supports the newer multi-source data mixture.
- The official [SmolLM2 alignment recipes](https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm2)
  use Smol-Smoltalk followed by UltraFeedback preference tuning for the small checkpoints.
- The [HelpSteer2 preference paper](https://arxiv.org/abs/2410.01257) describes its human comparison
  collection, Bradley–Terry reward modeling, and REINFORCE-based alignment experiments.
- TRL documents the current [`SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer),
  [`RewardTrainer`](https://huggingface.co/docs/trl/reward_trainer), and true online
  [`RLOOTrainer`](https://huggingface.co/docs/trl/rloo_trainer) used by these stages.
- The [FineWeb report](https://arxiv.org/abs/2406.17557) found large gains on knowledge and reasoning
  benchmarks from educational-quality filtering.
- [DataComp-LM](https://arxiv.org/abs/2406.11794) found model-based filtering to be central to strong
  open pretraining data.
- [Chinchilla](https://arxiv.org/abs/2203.15556) explains compute-optimal token/model scaling, while
  SmolLM's own ablations show small models can continue improving with overtraining. Disk capacity
  alone is not a training-budget argument.

Review the upstream dataset cards and licenses before publishing weights or using them commercially;
the mixture contains sources with distinct provenance and terms.



llm120-generate \
    --checkpoint /home/punam/Documents/120r/checkpoints/llm120-english/checkpoint-000049000 \
    --sample \
    --temperature 0.7 \
    --top-p 0.9 \
    --repetition-penalty 1.1 \
    --no-repeat-ngram-size 4 \
    "Explain photosynthesis."
