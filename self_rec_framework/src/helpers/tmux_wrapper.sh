#!/bin/bash
# General-purpose tmux wrapper for running long scripts in detached sessions
#
# Usage:
#   bash src/helpers/tmux_wrapper.sh <session_name> <command_to_run>
#
# Examples:
#   bash src/helpers/tmux_wrapper.sh data_gen "bash experiments/00_data_gen/bash/data_gen_sweep_pku.sh --batch"
#   bash src/helpers/tmux_wrapper.sh exp01 "bash experiments/01_AT_PW-C_Rec_Pr/bash/2_pku_sweep_other_models.sh --batch"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <session_name> <command_to_run>"
    echo ""
    echo "Examples:"
    echo "  $0 data_gen \"bash experiments/00_data_gen/bash/data_gen_sweep_pku.sh --batch\""
    echo "  $0 exp01 \"bash experiments/01_AT_PW-C_Rec_Pr/bash/2_pku_sweep_other_models.sh\""
    echo ""
    echo "Session Management:"
    echo "  View session:    tmux attach -t <session_name>"
    echo "  List sessions:   tmux ls"
    echo "  Kill session:    tmux kill-session -t <session_name>"
    echo "  Detach:          Ctrl+B then D"
    exit 1
fi

SESSION_NAME="$1"
shift
COMMAND="$*"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "✗ ERROR: tmux is not installed"
    echo "  Install with: sudo apt install tmux"
    exit 1
fi

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠ WARNING: Session '$SESSION_NAME' already exists"
    read -p "Kill and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$SESSION_NAME"
        echo "✓ Killed existing session"
    else
        echo "✗ Aborted. Use a different session name or attach with: tmux attach -t $SESSION_NAME"
        exit 1
    fi
fi

# Create new detached session
# The command runs in a login shell and keeps the session open after completion
tmux new-session -d -s "$SESSION_NAME" "cd $(pwd) && $COMMAND; echo ''; echo '=== Command completed. Press Ctrl+B then D to detach, or Ctrl+D to exit ==='; exec bash -l"

echo ""
echo "======================================================================="
echo "✓ TMUX SESSION STARTED"
echo "======================================================================="
echo "Session name: $SESSION_NAME"
echo "Command:      $COMMAND"
echo ""
echo "Management Commands:"
echo "  View output:     tmux attach -t $SESSION_NAME"
echo "  List sessions:   tmux ls"
echo "  Kill session:    tmux kill-session -t $SESSION_NAME"
echo ""
echo "Inside tmux:"
echo "  Detach (keep running):  Ctrl+B then D"
echo "  Exit (close session):   Ctrl+D or 'exit'"
echo "======================================================================="
echo ""
