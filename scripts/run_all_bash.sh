#!/bin/bash
# Run all bash scripts in a directory consecutively
#
# Usage: ./scripts/run_all_bash.sh <directory>
# Example: ./scripts/run_all_bash.sh experiments/ICML_01_UT_PW-Q_Rec_NPr_FA_Inst/bash
#
# Note: Continues running subsequent scripts even if one fails.
# Failed scripts are reported in the summary at the end.
# Auto-answers "y" to confirmation prompts (e.g., "Continue? (y/n):").

if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    echo "Example: $0 experiments/ICML_01_UT_PW-Q_Rec_NPr_FA_Inst/bash"
    exit 1
fi

DIR="$1"

if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' does not exist"
    exit 1
fi

# Find all .sh files in the directory, sorted alphabetically
SCRIPTS=$(find "$DIR" -maxdepth 1 -name "*.sh" -type f | sort)

if [ -z "$SCRIPTS" ]; then
    echo "No bash scripts found in '$DIR'"
    exit 0
fi

echo "======================================================================"
echo "Running all bash scripts in: $DIR"
echo "======================================================================"

# Count scripts and track results
TOTAL=$(echo "$SCRIPTS" | wc -l)
CURRENT=0
SUCCEEDED=0
FAILED=0
FAILED_SCRIPTS=""

for script in $SCRIPTS; do
    CURRENT=$((CURRENT + 1))
    SCRIPT_NAME=$(basename "$script")
    
    echo ""
    echo "======================================================================"
    echo "[$CURRENT/$TOTAL] Running: $SCRIPT_NAME"
    echo "======================================================================"
    echo ""
    
    # Run the script and capture exit code
    # Pipe "y" to auto-answer confirmation prompts (e.g., "Continue? (y/n):")
    if echo "y" | bash "$script"; then
        echo ""
        echo "✓ Completed: $SCRIPT_NAME"
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        EXIT_CODE=$?
        echo ""
        echo "✗ Failed: $SCRIPT_NAME (exit code: $EXIT_CODE)"
        FAILED=$((FAILED + 1))
        FAILED_SCRIPTS="$FAILED_SCRIPTS\n  ✗ $SCRIPT_NAME (exit code: $EXIT_CODE)"
    fi
done

echo ""
echo "======================================================================"
echo "SUMMARY"
echo "======================================================================"
echo "Total scripts: $TOTAL"
echo "Succeeded: $SUCCEEDED"
echo "Failed: $FAILED"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed scripts:"
    echo -e "$FAILED_SCRIPTS"
    echo ""
    echo "======================================================================"
    exit 1
else
    echo ""
    echo "All scripts completed successfully!"
    echo "======================================================================"
    exit 0
fi
