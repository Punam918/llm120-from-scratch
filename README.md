# LLM120: From-Scratch 125M Language Model

LLM120 is a compact, single-GPU training pipeline for building a decoder-only English language model from random initialization and a tokenizer trained from scratch. The project follows the practical design of the SmolLM2 family, but is adapted for a laptop-class setup and a reproducible local workflow.

This repository includes:

- dataset preparation and tokenizer training
- pretraining pipeline for a ~125M parameter model
- checkpointing and resume support
- generation and evaluation utilities
- post-training stages for SFT, DPO, reward modeling, and lightweight RLHF

## Why this project exists

The goal is to make a small but real language model training workflow understandable and executable without a large cluster. It is designed around a single 8 GB GPU workflow and focuses on reproducibility, diagnostics, and training discipline.

The default architecture is intentionally close to the SmolLM2-135M design, with a byte-level BPE tokenizer, 30 transformer layers, GQA attention, RMSNorm, RoPE, and tied embeddings. The project is meant to be a practical reference implementation and a useful experimental platform rather than a black-box benchmark runner.

## Features

- Train a decoder-only model from scratch
- Use a fresh tokenizer trained on the chosen corpus
- Support end-to-end pilot runs before full training
- Resume from checkpoints without losing optimizer state
- Run generation, benchmark, and post-training stages
- Keep the workflow compatible with `uv` and Python 3.11+

## Requirements

- Python 3.11+
- CUDA 12.8-compatible PyTorch setup for `sm_120` GPUs
- A machine with enough disk space for datasets, tokenized corpora, and checkpoints
- `uv` for dependency management

## Quick start

```bash
uv sync --extra dev --extra posttrain
source .venv/bin/activate
llm120-benchmark --vocab-size 32768 --steps 10
```

If `uv sync` reports that a newer PyTorch CUDA wheel is required for `sm_120`, update the explicit index in `pyproject.toml` rather than falling back to an older CUDA build.

## 1. Pilot run

Before the full training run, validate installation, tokenization, data preparation, and throughput with a smaller pilot:

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

This is a system-checking stage, not a final model run.

## 2. Prepare the production corpus

For a serious training run, start with a clean production dataset directory and train the final tokenizer from the full sample:

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

The pipeline is designed to be restartable. Shards are fingerprinted and skipped when completed, and the training code uses memory-mapped data processing to avoid loading the full corpus into RAM.

## 3. Train the base model

```bash
llm120-benchmark --steps 20
llm120-train
```

For the 400M configuration:

```bash
llm120-benchmark --config configs/train_400m.yaml --steps 20
llm120-train --config configs/train_400m.yaml --resume none
```

Checkpointing is automatic, and repeated invocations resume from the latest valid checkpoint with optimizer and RNG state.

## 4. Generate text

```bash
llm120-generate "Photosynthesis is the process by which"
```

Sampling mode is also supported:

```bash
llm120-generate \
  --sample \
  --temperature 0.8 \
  --top-p 0.95 \
  "In a small village near the mountains,"
```

This project intentionally treats the model as a base completion model, not a chat-ready assistant, until post-training is completed.

## 5. Post-training workflow

The repo supports the full small-model alignment pipeline:

```bash
llm120-download-posttrain
llm120-prepare-posttrain
llm120-sft
llm120-dpo
llm120-reward
llm120-rlhf
llm120-chat "What causes rain? Explain it to a ten-year-old."
```

The post-training stages are:

1. SFT for instruction following
2. DPO for preference learning
3. reward modeling from human comparison data
4. short online RLHF using RLOO

## Dataset and training notes

Important: do not start with an arbitrary old corpus snapshot unless you know exactly what you are doing. The project uses a modern, compact, practical dataset mixture rather than a huge old collection that is unnecessarily expensive for a laptop GPU.

The recommended production dataset is the SmolLM2-style 10B-token sample, which is much more suitable for a single-GPU setup than a full 600B-token cluster training regime.

In practice, the success of this project depends on:

- measuring real throughput before large runs
- validating the tokenizer and data pipeline early
- checking held-out loss and generation quality before post-training
- avoiding false assumptions that a base model is already a chat assistant

## Repository layout

```text
configs/           configuration files for pilot and training runs
data/              raw and tokenized datasets
artifacts/         tokenizer output and generated artifacts
checkpoints/       model checkpoints and tensorboard logs
src/llm120/        training, data, model, generation, and post-training code
tests/             test suite
README.md          project overview
LICENSE            MIT license
```

## References

This repo is inspired by and aligned with the design ideas from the SmolLM and SmolLM2 work, including:

- SmolLM training report
- SmolLM2 model configuration and data mix
- UltraFeedback and HelpSteer2 preference datasets
- TRL training libraries for SFT, DPO, reward modeling, and RLOO
- FineWeb and DataComp-LM research on data quality and filtering

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.
