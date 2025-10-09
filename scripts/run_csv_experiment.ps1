# Simple PowerShell runner for CSV-based 2T experiments
#
# Usage:
#   .\run_csv_experiment.ps1 data/experiments/pairwise/two-turn/typos/S1/claude-3-5-haiku-20241022.csv
#
# Or with article text:
#   .\run_csv_experiment.ps1 "path/to/data.csv" "Article text here..."

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$CsvPath,

    [Parameter(Mandatory=$false, Position=1)]
    [string]$ArticleText = "",

    [Parameter(Mandatory=$false)]
    [string]$Model = "anthropic/claude-3-5-haiku-20241022",

    [Parameter(Mandatory=$false)]
    [string]$Task = "two_turn_summary_recognition_csv"
)

# Check if CSV file exists
if (!(Test-Path $CsvPath)) {
    Write-Error "CSV file not found: $CsvPath"
    exit 1
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Running 2T Experiment from CSV" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "CSV: $CsvPath"
Write-Host "Model: $Model"
Write-Host "Task: $Task"
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Build inspect eval command
$cmd = "inspect eval protocols/pairwise/tasks.py@$Task --model $Model -T csv_path=$CsvPath"

if ($ArticleText) {
    $cmd += " -T article_text=`"$ArticleText`""
}

Write-Host "Command: $cmd" -ForegroundColor Yellow
Write-Host ""

# Execute
Invoke-Expression $cmd
