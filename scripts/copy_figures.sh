#!/bin/bash
# Copy latest analysis figures to ICML_figure folder for paper submission
#
# This script automatically finds the newest dated directory for each experiment
# and copies the specified plot files to the ICML_figure folder.
#
# To add more figures, simply add entries to the FIGURES array below.

set +e  # Don't exit on error - we want to process all figures

# Create output directory
OUTPUT_DIR="AIWILD_figures"
mkdir -p "$OUTPUT_DIR"

# Define figures to copy
# Format: "experiment_path|filename|output_name"
# experiment_path: path without the dated directory (e.g., "data/analysis/_aggregated_data/ICML_01_UT_PW-Q_Rec_NPr_FA_Inst")
# filename: name of the file to copy
# output_name: name to use in ICML_figure folder (optional, defaults to filename)
FIGURES=(
    "data/analysis/_aggregated_data/ICML_07_UT_PW-Q_Rec_NPr_FA_Rsn-Inst|aggregated_performance_grouped.png|figure_1a_ICML_07_aggregated_performance_grouped.png"
    "data/analysis/_aggregated_data/ICML_07_UT_PW-Q_Rec_NPr_FA_Rsn-Inst|rank_distance_grouped_bar_chart.png|figure_2a_ICML_07_rank_distance_grouped_bar_chart.png"
    "data/analysis/_aggregated_data/ICML_07_UT_PW-Q_Rec_NPr_FA_Rsn-Inst|performance_vs_arena_ranking.png|figure_2c_ICML_07_performance_vs_arena_ranking.png"
    "data/analysis/_aggregated_data/ICML_07_UT_PW-Q_Rec_NPr_FA_Rsn-Inst|rank_distance_aggregated.png|figure_2e_ICML_07_rank_distance_aggregated.png"
    "data/analysis/_aggregated_data/ICML_07_UT_PW-Q_Rec_NPr_FA_Rsn-Inst|rank_distance_filtered_evaluator_rank.png|figure_3a_ICML_07_rank_distance_filtered_evaluator_rank.png"
    "data/analysis/_aggregated_data/ICML_07_UT_PW-Q_Rec_NPr_FA_Rsn-Inst|rank_distance_filtered_evaluator_rank_positive.png|figure_3c_ICML_07_rank_distance_filtered_evaluator_rank_positive.png"

    
    "data/analysis/_aggregated_data/ICML_08_UT_IND-Q_Rec_NPr_FA_Rsn-Inst|aggregated_performance_grouped.png|figure_1b_ICML_08_aggregated_performance_grouped.png"
    "data/analysis/_aggregated_data/ICML_08_UT_IND-Q_Rec_NPr_FA_Rsn-Inst|rank_distance_grouped_bar_chart.png|figure_2b_ICML_08_rank_distance_grouped_bar_chart.png"
    "data/analysis/_aggregated_data/ICML_08_UT_IND-Q_Rec_NPr_FA_Rsn-Inst|performance_vs_arena_ranking.png|figure_2d_ICML_08_performance_vs_arena_ranking.png"
    "data/analysis/_aggregated_data/ICML_08_UT_IND-Q_Rec_NPr_FA_Rsn-Inst|rank_distance_adjusted.png|figure_2f_ICML_08_rank_distance_adjusted.png"
    "data/analysis/_aggregated_data/ICML_08_UT_IND-Q_Rec_NPr_FA_Rsn-Inst|rank_distance_filtered_evaluator_rank.png|figure_3b_ICML_08_rank_distance_filtered_evaluator_rank.png"
    "data/analysis/_aggregated_data/ICML_08_UT_IND-Q_Rec_NPr_FA_Rsn-Inst|rank_distance_filtered_evaluator_rank_positive.png|figure_3d_ICML_08_rank_distance_filtered_evaluator_rank_positive.png"
    "data/analysis/_aggregated_data/ICML_07-vs-ICML_08|performance_contrast_grouped.png|figure_1c_ICML_07_vs_ICML_08_performance_contrast_grouped.png"
    "data/analysis/_aggregated_data/ICML_01-vs-ICML_05|performance_scatter.png|figure_4a_ICML_01_vs_ICML_05_performance_scatter.png"
    "data/analysis/_aggregated_data/ICML_02-vs-ICML_06|performance_scatter.png|figure_4b_ICML_02_vs_ICML_06_performance_scatter.png"
)

echo "=========================================="
echo "Copying ICML Figures"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Function to find newest dated directory
find_newest_dir() {
    local base_path="$1"
    if [ ! -d "$base_path" ]; then
        echo ""  # Return empty if directory doesn't exist
        return
    fi
    
    # Find all dated directories (format: YYYYMMDD_HHMMSS)
    # Sort by modification time, newest first, take the first one
    ls -td "$base_path"/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9] 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo ""
}

# Process each figure
copied=0
failed=0

for entry in "${FIGURES[@]}"; do
    # Split by pipe character
    IFS='|' read -r experiment_path filename output_name <<< "$entry"
    
    # If output_name is empty, use filename
    if [ -z "$output_name" ]; then
        output_name="$filename"
    fi
    
    # Find newest dated directory
    newest_dir=$(find_newest_dir "$experiment_path")
    
    if [ -z "$newest_dir" ]; then
        echo "⚠  Skipping: $experiment_path (no dated directories found)"
        ((failed++))
        continue
    fi
    
    # Construct full source path
    source_file="$experiment_path/$newest_dir/$filename"
    dest_file="$OUTPUT_DIR/$output_name"
    
    # Check if source file exists
    if [ ! -f "$source_file" ]; then
        echo "⚠  Skipping: $source_file (file not found)"
        ((failed++))
        continue
    fi
    
    # Copy the file
    cp "$source_file" "$dest_file"
    echo "✓  Copied: $output_name (from $newest_dir)"
    ((copied++))
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Successfully copied: $copied"
echo "Failed/Skipped: $failed"
echo "Total: $((copied + failed))"
echo ""
echo "Figures saved to: $OUTPUT_DIR/"
