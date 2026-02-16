# Tmux Wrapper for Long-Running Jobs

This directory contains a tmux wrapper script for running long-running batch jobs that survive terminal disconnection.

## Quick Start

### Option 1: Use Pre-configured Wrapper Scripts

```bash
# Data generation with batch mode
bash experiments/00_data_gen/bash/tmux_data_gen_sweep_pku.sh

# Experiment sweep with batch mode
bash experiments/01_AT_PW-C_Rec_Pr/bash/tmux_pku_sweep_other_models.sh
```

### Option 2: Use Wrapper Directly

```bash
# General syntax
bash src/helpers/tmux_wrapper.sh <session_name> "<command_to_run>"

# Examples
bash src/helpers/tmux_wrapper.sh data_gen "bash experiments/00_data_gen/bash/data_gen_sweep_pku.sh --batch"
bash src/helpers/tmux_wrapper.sh exp01 "bash experiments/01_AT_PW-C_Rec_Pr/bash/2_pku_sweep_other_models.sh --batch"
```

## Tmux Session Management

### View Running Session
```bash
# Attach to see output
tmux attach -t data_gen

# Inside session: Ctrl+B then D to detach (keeps running)
```

### List All Sessions
```bash
tmux ls
```

### Kill Session
```bash
tmux kill-session -t data_gen
```

## Why Use Tmux?

### Without Tmux:
- ❌ Close terminal = process dies
- ❌ SSH disconnect = process dies
- ❌ Lost output if connection drops

### With Tmux:
- ✅ Close terminal = process keeps running
- ✅ SSH disconnect = process keeps running
- ✅ Can reconnect anytime to check progress
- ✅ Perfect for batch mode (which takes hours)

## Batch Mode + Tmux Workflow

```bash
# 1. Start batch job in tmux
bash src/helpers/tmux_wrapper.sh data_gen "bash experiments/00_data_gen/bash/data_gen_sweep_pku.sh --batch"

# 2. Close terminal/SSH (safe to disconnect)

# 3. Later, check progress
tmux attach -t data_gen

# 4. Detach again
# Press: Ctrl+B then D

# 5. When done, session stays open for review
tmux attach -t data_gen  # View final output
tmux kill-session -t data_gen  # Clean up when done
```

## Creating Custom Wrapper Scripts

Create a wrapper for any long-running script:

```bash
#!/bin/bash
# experiments/my_experiment/bash/tmux_my_task.sh

bash src/helpers/tmux_wrapper.sh \
    my_session_name \
    "bash path/to/my_script.sh --batch --other-flags"
```

Make it executable:
```bash
chmod +x experiments/my_experiment/bash/tmux_my_task.sh
```
