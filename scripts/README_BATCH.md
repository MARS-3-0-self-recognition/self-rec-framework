# Batch Mode Tools & Best Practices

This directory contains utilities for managing batch API jobs across providers (OpenAI, Anthropic, Google, Together).

## 🚨 CRITICAL: Always Use tmux for Batch Runs

**Batch jobs continue running on provider servers even after you close your terminal.**

### **The Problem:**

When you use `--batch` mode and close your terminal:
- ✅ Jobs are submitted to provider's batch queue
- ✅ Jobs run on provider servers (may complete successfully)
- ❌ **Results are LOST** - no way to retrieve them
- ⚠️ **Still charged** for completed batches

### **The ONLY Solution:**

**Use tmux** - it's the only reliable way to run batch experiments:

```bash
# Use existing tmux wrappers
bash experiments/12_UT_PW-Q_Rec_Pr/bash/tmux_pku_sweep_other_models.sh

# Or manually:
tmux new -s my_sweep
bash experiments/12_UT_PW-Q_Rec_Pr/bash/2_pku_sweep_other_models.sh
# Ctrl+B then D to detach

# Reconnect later:
tmux attach -t my_sweep
```

**Why tmux is necessary:**
- Inspect AI doesn't store batch IDs in eval logs
- Can't reconnect to orphaned batches
- Can't retrieve results without active process
- tmux keeps the process alive even if you close terminal/disconnect

---

## 📋 Available Tools

### 1. `cancel_all_batches.py`
Cancel all active batch jobs across all providers.

**When to use:**
- After accidentally closing terminal during batch run
- To clean up orphaned batch jobs
- Before re-running experiments

```bash
uv run python scripts/cancel_all_batches.py
```

**Output:**
```
OpenAI: ✓ Cancelled 13 batches
Anthropic: ✓ Cancelled 14 batches
SUMMARY: Cancelled 27 total batch jobs
```

---

### 2. `list_active_batches.py`
List all active batch jobs and optionally save their IDs.

**Basic usage:**
```bash
uv run python scripts/list_active_batches.py
```

**Save batch IDs for later:**
```bash
uv run python scripts/list_active_batches.py \
    --save data/results/my_experiment/batch_tracking.json
```

---

### 3. `analyze_pairwise_results.py`
Analyze completed pairwise self-recognition experiments.

**Usage:**
```bash
uv run python scripts/analyze_pairwise_results.py \
    --results_dir data/results/pku_saferlhf/mismatch_1-20/12_UT_PW-Q_Rec_Pr
```

**Outputs:**
- `data/analysis/.../accuracy_pivot.csv`: Raw accuracy matrix
- `data/analysis/.../accuracy_heatmap.png`: Visualization
- `data/analysis/.../summary_stats.txt`: Statistics

---

## ✅ Best Practices

### ALWAYS Use tmux for Batch Runs

**This is not optional.** It's the **only** way to safely run batch experiments:

```bash
# Use existing tmux wrappers (recommended)
bash experiments/12_UT_PW-Q_Rec_Pr/bash/tmux_pku_sweep_other_models.sh
```

**Why tmux is mandatory:**
- ✅ Process stays alive even if terminal closes
- ✅ Reconnect anytime with `tmux attach`
- ✅ See real-time progress
- ✅ Results are properly retrieved
- ❌ **No alternative exists** - batch IDs can't be recovered

**If you don't use tmux:**
- Your batch jobs will be orphaned
- Results will be lost
- You'll waste API credits
- You'll have to cancel and re-run everything

---

## 🔧 Recovery from Interrupted Batch Runs

If you accidentally closed your terminal during a batch run:

### **Bad News:**
- **Results are permanently lost** - no way to retrieve them
- Batch jobs may still be running on provider servers
- You're wasting credits on jobs you can't use

### **Steps to Recover:**

#### 1. Check Active Batches
```bash
uv run python scripts/list_active_batches.py
```

#### 2. Cancel Orphaned Batches
```bash
# Cancel all active batches to stop wasting credits
uv run python scripts/cancel_all_batches.py
```

#### 3. Re-run with tmux
```bash
# THIS TIME, use tmux!
bash experiments/12_UT_PW-Q_Rec_Pr/bash/tmux_pku_sweep_other_models.sh
```

**The sweep script automatically:**
- ✅ Skips "started" status (won't duplicate orphaned jobs)
- ✅ Retries "error"/"cancelled" evaluations
- ✅ Skips successful evaluations

---

## 🐛 Known Issues

### Batch ID Recovery (Fundamental Limitation)
**Issue:** Inspect AI doesn't store batch IDs in eval logs
**Why:** Batch IDs are internal to provider implementations, never exposed
**Impact:** Can't recover results from interrupted batch runs
**Workaround:** **Use tmux** - only reliable solution
**Status:** Unfixable without changes to Inspect AI core

### Google Gemini Batch Mode
**Issue:** Google batch API has bugs in Inspect AI
**Workaround:** Sweep script automatically disables batch for Gemini evaluators
**Status:** Shows warning in red before running

### Together AI Logprobs
**Issue:** qwen-3.0-80b and deepseek-3.1 crash with logprobs
**Workaround:** Logprobs disabled by default (use `--logprobs` to enable and get NotImplementedError)
**Status:** Fixed - no crashes

---

## 📊 Analysis Workflow

After experiments complete:

```bash
# 1. Analyze results
uv run python scripts/analyze_pairwise_results.py \
    --results_dir data/results/pku_saferlhf/mismatch_1-20/12_UT_PW-Q_Rec_Pr

# 2. View outputs
ls data/analysis/pku_saferlhf/mismatch_1-20/12_UT_PW-Q_Rec_Pr/
#   accuracy_pivot.csv
#   accuracy_heatmap.png
#   summary_stats.txt
```

---

## 💡 Critical Reminders

1. **🔴 ALWAYS USE TMUX FOR BATCH RUNS** - Not optional, not negotiable
2. **Check active batches before re-running:** `scripts/list_active_batches.py`
3. **Cancel orphaned batches:** `scripts/cancel_all_batches.py` if you accidentally interrupted
4. **Review confirmation prompt:** Sweep shows exactly what will run before executing
5. **Gemini + batch = automatic split:** Non-Gemini with batch, Gemini without
6. **"Started" status = skip:** Script won't re-run potentially active batch jobs

---

## 🎯 Quick Start

**To run a batch sweep correctly:**

```bash
# 1. Use tmux wrapper (handles everything)
bash experiments/12_UT_PW-Q_Rec_Pr/bash/tmux_pku_sweep_other_models.sh

# 2. Detach safely (Ctrl+B then D)

# 3. Check progress anytime
tmux attach -t pku_sweep_other_models

# 4. After completion, analyze
uv run python scripts/analyze_pairwise_results.py \
    --results_dir data/results/pku_saferlhf/mismatch_1-20/12_UT_PW-Q_Rec_Pr
```

**That's it. Don't overcomplicate it.**
