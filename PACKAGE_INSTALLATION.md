# Package Installation Guide

This guide explains how to install and use `self-rec-framework` as an importable package in other projects.

## Installation Options

### Option 1: Editable Install (Development)

For development or when you want changes to be immediately available:

```bash
# From the self-rec-framework directory
cd /path/to/self-rec-framework

# Install in editable mode using uv
uv pip install -e .

# Or using pip
pip install -e .
```

This installs the package in "editable" mode, meaning changes to the source code are immediately reflected without reinstalling.

### Option 2: Install from Local Directory

If you want to install it as a regular package (not editable):

```bash
# From the self-rec-framework directory
cd /path/to/self-rec-framework

# Build and install
uv pip install .

# Or using pip
pip install .
```

### Option 3: Install from Git Repository

Install directly from the Git repository:

```bash
pip install git+https://github.com/your-org/self-rec-framework.git

# Or for a specific branch/tag
pip install git+https://github.com/your-org/self-rec-framework.git@branch-name
```

### Option 4: Install from PyPI (if published)

If the package is published to PyPI:

```bash
pip install self-rec-framework
```

## Usage in Other Projects

Once installed, you can import modules from the package:

```python
# Import helpers
from self_rec_framework.src.helpers.model_names import (
    inspect_model_name,
    short_model_name,
    LM_ARENA_RANKINGS,
)
from self_rec_framework.src.helpers.model_sets import get_model_set
from self_rec_framework.src.helpers.constants import MY_DATASET_NAMESPACE

# Import data loading utilities
from self_rec_framework.src.data_generation.data_loading import (
    load_wikisum,
    load_pku_saferlhf,
    load_bigcodebench,
    load_sharegpt,
)

# Import procedural editing utilities
from self_rec_framework.src.data_generation.procedural_editing import (
    caps,
    typos,
    treatment,
)

# Import Inspect AI integration
from self_rec_framework.src.inspect.tasks import get_task_function
from self_rec_framework.src.inspect.config import load_experiment_config
from self_rec_framework.src.inspect.scorer import logprob_scorer
```

**Note:** Python package names cannot contain hyphens, so the import uses `self_rec_framework` (with underscores) even though the PyPI package name is `self-rec-framework` (with hyphens).

## Package Structure

The package exposes the following modules:

- `self_rec_framework.src.helpers` - Utility functions for model names, sets, constants
- `self_rec_framework.src.data_generation` - Data loading and procedural editing
- `self_rec_framework.src.inspect` - Inspect AI integration for evaluation
- `self_rec_framework.src.core_prompts` - Core prompt configurations

## Verifying Installation

To verify the package is installed correctly:

```python
import self_rec_framework.src.helpers.model_names
print(self_rec_framework.src.helpers.model_names.__file__)  # Should show the installed location
```

Or test an import:

```python
from self_rec_framework.src.helpers.model_names import inspect_model_name
print(inspect_model_name("gpt-4o-mini"))  # Should work
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'self_rec_framework'`:

1. Make sure the package is installed: `pip list | grep self-rec-framework`
2. Check your Python environment matches where you installed the package
3. Try reinstalling: `pip install -e .` (from the package directory)
4. Remember: use `self_rec_framework` (underscores) in imports, not `self-rec-framework` (hyphens)

### Path Issues

If imports work but you get path-related errors:

- The package uses relative paths for data/config files
- Make sure you're running from the correct working directory
- Or set environment variables if the package expects specific paths

### Development Workflow

For active development:

1. Install in editable mode: `pip install -e .`
2. Make changes to source code
3. Changes are immediately available (no reinstall needed)
4. Run tests to verify changes

## Building a Distribution Package

To create a distributable package:

```bash
# Build wheel and source distribution
python -m build

# This creates files in dist/
# dist/self_rec_framework-0.1.0-py3-none-any.whl
# dist/self-rec-framework-0.1.0.tar.gz
```

These can be shared or uploaded to PyPI.
