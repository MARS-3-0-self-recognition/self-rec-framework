# Procedural Editing

This directory contains scripts for applying procedural treatments to model outputs in JSON format. These treatments are used to test how different modifications affect model self-recognition capabilities.

## Scripts

### `apply_caps_treatment.py`

Applies capitalization treatments to model outputs with configurable strength levels.

**Usage:**
```bash
# Apply S2 treatment (50% caps) to simple_config.json
uv run src/data_gen/procedural_editing/apply_caps_treatment.py \
    --input_file=data/cnn_debug/3-5-sonnet/simple_config.json \
    --treatment_name=caps_s2 \
    --strength=S2

# Apply S4 treatment (100% caps) with custom output directory
uv run src/data_gen/procedural_editing/apply_caps_treatment.py \
    --input_file=data/wikisum_debug/input.json \
    --treatment_name=caps_s4 \
    --strength=S4 \
    --output_dir=data/wikisum_debug
```

**Arguments:**
- `--input_file`: Path to input JSON file (UUID->text format)
- `--treatment_name`: Name for the treatment (e.g., 'caps_s1', 'caps_s2')
- `--strength`: Treatment strength (S1, S2, S3, S4) - mutually exclusive with --percentage
- `--percentage`: Custom capitalization percentage (0-100) - mutually exclusive with --strength
- `--output_dir`: Output directory (default: same as input file directory)
- `--seed`: Random seed for reproducible results

**Note:** Either `--strength` or `--percentage` must be specified, but not both.

**Strength Levels:**
- **S1**: 25% capitalization (light)
- **S2**: 50% capitalization (medium)
- **S3**: 75% capitalization (heavy)
- **S4**: 100% capitalization (all caps)

### `apply_typo_treatment.py`

Applies typo treatments to model outputs using realistic keyboard-based substitutions and character deletions.

**Usage:**
```bash
# Apply S1 treatment (0.1 typos per word) to simple_config.json
uv run src/data_gen/procedural_editing/apply_typo_treatment.py \
    --input_file=data/cnn_debug/3-5-sonnet/simple_config.json \
    --treatment_name=typo_s1 \
    --strength=S1

# Apply S3 treatment (0.6 typos per word) with seed for reproducibility
uv run src/data_gen/procedural_editing/apply_typo_treatment.py \
    --input_file=data/wikisum_debug/input.json \
    --treatment_name=typo_s3 \
    --strength=S3 \
    --seed=42
```

**Arguments:**
- `--input_file`: Path to input JSON file (UUID->text format)
- `--treatment_name`: Name for the treatment (e.g., 'typo_s1', 'typo_s2')
- `--strength`: Treatment strength (S1, S2, S3, S4) - mutually exclusive with --custom
- `--custom`: Use custom parameters (requires all custom rate parameters)
- `--typos_per_word`: Custom typos per word rate (>=0) - required when using --custom
- `--flip_rate`: Percentage of adjacent letter pairs to flip (0-100) - required when using --custom
- `--drop_rate`: Percentage of characters to drop (0-100) - required when using --custom
- `--add_rate`: Percentage of positions to add random characters (0-100) - required when using --custom
- `--substitute_rate`: Percentage of characters to substitute (0-100) - required when using --custom
- `--output_dir`: Output directory (default: same as input file directory)
- `--seed`: Random seed for reproducible results

**Note:** Either `--strength` or `--custom` must be specified, but not both. When using `--custom`, all rate parameters must be provided.

**Strength Levels:**
- **S1**: 0.1 typos per word (light)
- **S2**: 0.3 typos per word (medium)
- **S3**: 0.6 typos per word (heavy)
- **S4**: 1.2 typos per word (major)

## Input Format

Both scripts expect JSON files with UUID->text mappings:

```json
{
  "06352019a19ae31e527f37f7571c6dd7f0c5da37": "A bridge suddenly collapsed, causing vehicles to fall into a river...",
  "24521a2abb2e1f5e34e6824e0f9e56904a2b0e88": "President Bush underwent a colonoscopy, during which 5 small polyps...",
  "ee8871b15c50d0db17b0179a6d2beab35065f1e9": "A Miami jail is housing many mentally ill inmates..."
}
```

## Output Format

Scripts generate new JSON files named `{treatment_name}_config.json` in the same directory as the input file (or specified output directory):

```json
{
  "06352019a19ae31e527f37f7571c6dd7f0c5da37": "a BRIdge SUDDeNLy cOllApsED, cAUSing vehIcles to FALL INTo A RIVer...",
  "24521a2abb2e1f5e34e6824e0f9e56904a2b0e88": "presIDENT BuSh UNderwENt a coLonoScOPy, dURiNg wHich 5 SmALl POlYPs...",
  "ee8871b15c50d0db17b0179a6d2beab35065f1e9": "A MIAmi JAil Is hoUSING mANy MeNTaLly ILL inMAtEs..."
}
```

## Treatment Details

### Capitalization Treatment

- Randomly capitalizes letters based on the specified percentage
- Preserves word boundaries and punctuation
- S4 treatment converts all letters to uppercase

### Typo Treatment

The typo treatment supports two different approaches:

**Strength-based (--strength):**
- Uses per-word typo introduction with configurable rates
- Applies exactly one typo type per word when selected (randomly chosen from available types)
- More controlled, word-level modifications with even distribution
- Examples: S1 (0.1 typos/word), S2 (0.3 typos/word), S3 (0.6 typos/word), S4 (1.2 typos/word)

**Custom parameters (--custom):**
- Uses direct character-level rates across the entire text
- Applies modifications in order: drops, substitutions, flips, additions
- More aggressive, character-level modifications
- Allows precise control over each typo type:
  - `--flip_rate`: Percentage of adjacent letter pairs to flip (0-100)
  - `--drop_rate`: Percentage of characters to drop (0-100)
  - `--add_rate`: Percentage of positions to add random characters (0-100)
  - `--substitute_rate`: Percentage of characters to substitute (0-100)

**Common features:**
- Uses realistic keyboard-based substitutions based on QWERTY layout
- Skips very short words (1-2 characters) to avoid breaking them
- Examples of realistic substitutions:
  - `q` → `w` or `a`
  - `e` → `w`, `r`, `s`, or `d`
  - `1` → `2` or `q`

## Examples

### Basic Usage

```bash
# Apply light caps treatment
uv run src/data_gen/procedural_editing/apply_caps_treatment.py \
    --input_file=data/cnn_debug/3-5-sonnet/simple_config.json \
    --treatment_name=caps_light \
    --strength=S1

# Apply heavy typo treatment
uv run src/data_gen/procedural_editing/apply_typo_treatment.py \
    --input_file=data/wikisum_debug/input.json \
    --treatment_name=typo_heavy \
    --strength=S3
```

### Custom Parameters

```bash
# Apply custom caps treatment (15% capitalization)
uv run src/data_gen/procedural_editing/apply_caps_treatment.py \
    --input_file=data/cnn_debug/3-5-sonnet/simple_config.json \
    --treatment_name=caps_custom \
    --percentage=15

# Apply custom typo treatment with individual rates
uv run src/data_gen/procedural_editing/apply_typo_treatment.py \
    --input_file=data/wikisum_debug/input.json \
    --treatment_name=typo_custom \
    --custom \
    --typos_per_word=0.5 \
    --flip_rate=10 \
    --drop_rate=5 \
    --add_rate=3 \
    --substitute_rate=15
```

### Batch Processing

```bash
# Process multiple datasets with different treatments
for dataset in data/*/input.json; do
    echo "Processing $dataset"

    # Apply caps treatments
    uv run src/data_gen/procedural_editing/apply_caps_treatment.py \
        --input_file="$dataset" \
        --treatment_name=caps_s2 \
        --strength=S2

    # Apply typo treatments
    uv run src/data_gen/procedural_editing/apply_typo_treatment.py \
        --input_file="$dataset" \
        --treatment_name=typo_s2 \
        --strength=S2
done
```

### Reproducible Results

```bash
# Use seed for consistent results across runs
uv run src/data_gen/procedural_editing/apply_caps_treatment.py \
    --input_file=data/cnn_debug/3-5-sonnet/simple_config.json \
    --treatment_name=caps_reproducible \
    --strength=S2 \
    --seed=42
```

## Integration with Self-Recognition Framework

These treatment scripts are designed to work with the self-recognition framework's data generation pipeline:

1. **Load datasets** using scripts in `data_loader/`
2. **Apply treatments** using these procedural editing scripts
3. **Generate model outputs** using `src/data_gen/gen.py`
4. **Test self-recognition** using protocols in `src/protocols/`

The treated outputs can be used to test how different types of modifications affect a model's ability to recognize its own generated text.
