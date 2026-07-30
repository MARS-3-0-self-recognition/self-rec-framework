#!/bin/bash
# Run all bash scripts in one or more directories consecutively
#
# Usage: ./scripts/utils/run.sh <directory1> [directory2] ...
# Example: ./scripts/utils/run.sh experiments/_tutorials/_T01_tutorial_evals_individual/bash/b00_analysis/_inter-dataset
#
# Note: Continues running subsequent scripts even if one fails.
# Failed scripts are reported in the summary at the end.
# Sweep scripts pass -y to run_experiment_sweep to skip confirmation. Do NOT
# pipe input (e.g. "echo y |") into the scripts: that makes stdin a pipe
# instead of the TTY and breaks Inspect's experiment GUI (mouse/keyboard).

if [ -z "$1" ]; then
    echo "Usage: $0 <directory1> [directory2] ..."
    echo "Example: $0 experiments/_tutorials/_T01_tutorial_evals_individual/bash/b00_analysis/_inter-dataset"
    exit 1
fi

# Enable batch mode for all sweep scripts via environment variable
export SWEEP_BATCH=0

# Global counters
TOTAL_ALL=0
SUCCEEDED_ALL=0
FAILED_ALL=0
FAILED_SCRIPTS_ALL=""

# Iterate over all provided directories
for DIR in "$@"; do
    if [ ! -d "$DIR" ]; then
        echo "Error: Directory '$DIR' does not exist. Skipping."
        continue
    fi

    # Find all .sh files in the directory, sorted alphabetically
    # Exclude config.sh (shared config, sourced not run) and run_*.sh entry
    # points (run_sweep.sh / run_all_analyses.sh etc.) — those invoke THIS
    # runner, so executing them here would recurse or double-run.
    # Collect .sh scripts via a sorted glob (alphabetical, and space-safe:
    # pathname expansion is NOT subject to word-splitting), excluding config.sh
    # (sourced, not run) and run_*.sh entry points (they invoke this runner).
    # This replaces `for script in $(find ...)`, which word-splits the paths on
    # any space in the directory (e.g. ".../My Project/..."). Bash-array syntax
    # used here is portable back to bash 3.2 (the default /bin/bash on macOS).
    SCRIPTS=()
    for script in "$DIR"/*.sh; do
        [ -e "$script" ] || continue          # no matches -> literal glob, skip
        case "$(basename "$script")" in
            config.sh|run_*.sh) continue ;;
        esac
        SCRIPTS+=("$script")
    done

    if [ ${#SCRIPTS[@]} -eq 0 ]; then
        echo "No bash scripts found in '$DIR'"
        continue
    fi

    echo "======================================================================"
    echo "Running all bash scripts in: $DIR"
    echo "======================================================================"

    # Count scripts in current directory
    TOTAL_DIR=${#SCRIPTS[@]}
    CURRENT_DIR=0

    for script in "${SCRIPTS[@]}"; do
        CURRENT_DIR=$((CURRENT_DIR + 1))
        TOTAL_ALL=$((TOTAL_ALL + 1))
        
        SCRIPT_NAME=$(basename "$script")

        echo ""
        echo "======================================================================"
        echo "[$CURRENT_DIR/$TOTAL_DIR in $DIR] Running: $SCRIPT_NAME"
        echo "======================================================================"
        echo ""

        # Run the script and capture exit code
        if bash "$script"; then
            echo ""
            echo "✓ Completed: $SCRIPT_NAME"
            SUCCEEDED_ALL=$((SUCCEEDED_ALL + 1))
        else
            EXIT_CODE=$?
            echo ""
            echo "✗ Failed: $SCRIPT_NAME (exit code: $EXIT_CODE)"
            FAILED_ALL=$((FAILED_ALL + 1))
            FAILED_SCRIPTS_ALL="$FAILED_SCRIPTS_ALL\n  ✗ $DIR/$SCRIPT_NAME (exit code: $EXIT_CODE)"
        fi
    done
    echo ""
done

echo ""
echo "======================================================================"
echo "GLOBAL SUMMARY"
echo "======================================================================"
echo "Total scripts: $TOTAL_ALL"
echo "Succeeded: $SUCCEEDED_ALL"
echo "Failed: $FAILED_ALL"

if [ $FAILED_ALL -gt 0 ]; then
    echo ""
    echo "Failed scripts:"
    echo -e "$FAILED_SCRIPTS_ALL"
    echo ""
    echo "======================================================================"
    exit 1
else
    if [ $TOTAL_ALL -eq 0 ]; then
        echo ""
        echo "No scripts were found to run."
        echo "======================================================================"
        exit 0
    else
        echo ""
        echo "All scripts completed successfully!"
        echo "======================================================================"
        exit 0
    fi
fi
