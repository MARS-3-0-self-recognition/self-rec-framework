"""CSV-based data loading utilities for pairwise self-recognition tasks."""

import pandas as pd
from typing import List, Dict, Any


def load_dataset_from_csv(
    csv_path: str,
    article_text: str | None = None,
    trial_col: str = "trial",
    source1_col: str = "source1",
    content1_col: str = "content1",
    source2_col: str = "source2",
    content2_col: str = "content2",
) -> List[Dict[str, Any]]:
    """
    Load pairwise comparison dataset from a CSV file.

    The CSV should have columns for trial, source models, and content from each source.
    This creates TWO samples per row - one with each ordering (source1-first, source2-first).

    Args:
        csv_path: Path to the CSV file
        article_text: Optional article text (if all trials use the same article)
        trial_col: Name of trial column (default: "trial")
        source1_col: Name of first source model column (default: "source1")
        content1_col: Name of first content column (default: "content1")
        source2_col: Name of second source model column (default: "source2")
        content2_col: Name of second content column (default: "content2")

    Returns:
        List of sample dictionaries (2 per row) containing:
        - content: The article/question text (if provided, otherwise empty string)
        - output1: First output
        - output2: Second output
        - metadata: Dict with correct_answer, trial, sources, and ordering info
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Filter out empty rows
    df = df.dropna(subset=[trial_col])

    # Create TWO samples per row - one for each ordering
    samples = []

    for idx, row in df.iterrows():
        trial = row[trial_col]
        source1_name = str(row[source1_col])
        content1 = str(row[content1_col])
        source2_name = str(row[source2_col])
        content2 = str(row[content2_col])

        # Skip if either content is empty/NaN
        if pd.isna(content1) or pd.isna(content2) or not content1 or not content2:
            continue

        metadata_base = {
            "trial": trial,
            "csv_row": idx,
            "base_model": source1_name,
            "alternative_model": source2_name,
        }

        # Sample 1: source1 output first (position 1) - correct answer is "1"
        samples.append(
            {
                "content": article_text or "",  # Article text if provided
                "output1": content1,  # Base model output (correct)
                "output2": content2,  # Treatment output
                "metadata": {
                    **metadata_base,
                    "correct_answer": "1",
                    "ordering": "base_first",
                },
            }
        )

        # Sample 2: source2 output first (position 1) - correct answer is "2"
        samples.append(
            {
                "content": article_text or "",
                "output1": content2,  # Treatment output
                "output2": content1,  # Base model output (correct)
                "metadata": {
                    **metadata_base,
                    "correct_answer": "2",
                    "ordering": "base_second",
                },
            }
        )

    print(f"Loaded {len(df)} rows from CSV, created {len(samples)} samples (2 per row)")

    return samples
