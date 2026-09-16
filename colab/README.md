# LLM120 on Google Colab (free GPU)

A self-contained package for running the **LLM120 from-scratch pretraining pipeline** on
Google Colab's free GPU — **download → tokenizer → tokenize → train → generate** — for the
**300M** or **400M** model, with **checkpoints saved to your Google Drive** so a run survives
Colab disconnects and resumes across sessions.

## What's in this folder

```
colab/
├── LLM120_Colab.ipynb      ← open this in Colab (the whole pipeline, one cell per stage)
├── src/llm120/             ← the project's Python code (a copy of the repo's src/)
├── requirements-colab.txt  ← pip deps (PyTorch is NOT listed — Colab provides it)
└── README.md               ← this file
```

Upload the **entire `colab/` folder** to your Google Drive (e.g. to `My Drive/colab`), then in
Drive right-click `LLM120_Colab.ipynb` → **Open with → Google Colaboratory**.

## Quick start

1. **Upload** the `colab/` folder to Google Drive.
2. Open `LLM120_Colab.ipynb` from Drive.
3. **Runtime → Change runtime type → T4 GPU.**
4. Edit the **settings cell** at the top if you want (model size, token budget), then
   **Run All** (or run cells top to bottom).

The first session downloads data, trains the tokenizer, tokenizes, then trains. Later sessions
skip the data build and **resume training** from the newest Drive checkpoint.

## The free-tier reality — read this before committing to a big run

| Resource | Free Colab / Drive |
|---|---|
| GPU | **T4**, 16 GB VRAM (L4 only if you're lucky) |
| Colab disk | ~100 GB **ephemeral** (wiped on disconnect) |
| System RAM | ~12 GB |
| Max session | **~12 h**, then **~12 h cooldown** |
| Idle disconnect | **~90 min** of inactivity |
| Google Drive | **15 GB** total, shared with Gmail + Photos |

Two hard facts shape everything here:

1. **The full 10B-token run is not realistic on the free tier.** It is weeks of continuous GPU
   time, and the 20 GB tokenized dataset does not fit the 15 GB free Drive. The notebook
   therefore defaults to a **~1B-token budget** that actually completes (a few sessions) and fits
   in Drive. Raise it with `MAX_TOKENS` if you have the time/storage.
2. **The T4 has no BF16.** The trainer aborts if it can't get BF16, so the notebook detects this
   and auto-selects **FP16 mixed precision** (which the trainer already supports). On an L4 or
   A100 it uses BF16 automatically. **Do not** install the repo's `cu128` PyTorch index — that is
   for the author's Blackwell (sm_120) laptop GPU and is wrong for Colab's T4 (sm_75); this
   package skips it and uses Colab's preinstalled torch.

The good news: the trainer's **resume** is exact — it restores weights, optimizer, gradient
scaler, RNG state, and the precise data block. The only loss on a disconnect is the progress
since the last periodic checkpoint.

## Settings (top cell of the notebook)

| Setting | Default | Notes |
|---|---|---|
| `MODEL_SIZE` | `"400m"` | `"300m"` trains ~25% faster for the same budget. |
| `MAX_TOKENS` | `1_000_000_000` (1B) | ≈2 GB tokenized data. Try 2–3B later. 10B needs paid storage. |
| `MICRO_BATCH` | `2` | T4 has 16 GB. Raise to `4` for speed; drop to `1` if you OOM. Global batch stays 65,536 tokens. |
| `SAVE_EVERY_STEPS` | `250` | How often to checkpoint to Drive. Lower = safer on Colab. |
| `KEEP_LAST` | `2` | Rotating checkpoints kept on Drive. Each ≈4.8 GB for 400M (≈3.6 GB for 300M). |
| `DRIVE_PROJECT` | `/content/drive/MyDrive/colab` | Where you uploaded the folder. Change if you renamed it. |

For a **1B / 400M** run with `KEEP_LAST=2`: ~9.6 GB checkpoints + ~2 GB data ≈ **12 GB**, within
the 15 GB Drive limit. For a **2–3B** corpus, set `KEEP_LAST=1`. The **10B** corpus (20 GB) does
not fit free Drive at all — see *Scaling up*.

## Resuming after a disconnect (the normal workflow)

Colab will disconnect you (~12 h cap, or ~90 min idle). That's expected — the checkpoint on Drive
is your save point.

1. New session → **Runtime → T4 GPU**.
2. Run the **settings** cell **unchanged**, then **§1 Mount**, **§3 Data** (copies the cached
   corpus back to the VM), **§4 Config + doctor**, then **§6 Train**.
3. It resumes at the exact next optimizer step.

To avoid the idle disconnect, keep the browser tab focused, or paste the auto-reconnect snippet
(Developer-Tools console, F12) from the bottom of the notebook while training.

> **Don't change settings between sessions.** The resume check refuses a config whose SHA-256
> differs from the checkpoint's. To change the run, point `output_dir` at a new checkpoint folder
> (or use `--resume none`) and start fresh.

## Tips

- **Skip BPE training** by copying your local repo's `artifacts/tokenizer-32k/` into
  `colab/artifacts/tokenizer-32k/` on Drive *before* the first data build. The tokenizer must
  match the data, so only do this before tokenizing.
- **Benchmark first** (§5). It prints measured tokens/s and a 10B-token time estimate; scale that
  to your `MAX_TOKENS` to know how many sessions to expect.
- **Throughput** on a T4 for a 400M model is roughly ~1.5–3k tokens/s with FP16 + activation
  checkpointing, so 1B tokens is on the order of ~5–8 GPU-days → several sessions. 300M is faster.

## Scaling up beyond the free tier

If you want the **full 10B-token / 400M** run:

- **Colab Pro / Pay-As-You-Go** for longer sessions and L4/A100 GPUs (BF16, much faster), and/or
- **Paid Google Drive / Google One** (100 GB+) so the 20 GB dataset and several checkpoints fit.

The notebook's logic works unchanged — just raise `MAX_TOKENS`, and on a bf16-capable GPU it will
auto-use BF16.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA is unavailable` | Runtime → Change runtime type → T4 GPU. |
| **OOM / out of memory** | Lower `MICRO_BATCH` (4 → 2 → 1). Global batch stays the same. |
| Non-finite loss (NaN) | FP16 from scratch can spike. Re-run; if it repeats, lower `learning_rate` in the config cell. |
| Dataset download 401/auth | Paste a Hugging Face token into `HF_TOKEN`. |
| `Resume refused: ... differs` | You changed a setting mid-run. Use a new `output_dir` or `--resume none`. |
| Markdown/source looks wrong | Make sure you opened the `.ipynb` in Colab, not a text editor. |

## Scope and licensing

This package covers **pretraining + generation** (a base completion model). The repo's later
assistant stages (SFT / DPO / Reward / RLHF) are not wired up here — add them after you have a
coherent base model.

The corpus is `EleutherAI/SmolLM2-135M-10B`. Review that dataset's card and license before
publishing any weights. This is an independent training harness; it is not affiliated with the
SmolLM authors.
