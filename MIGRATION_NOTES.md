# Migration Notes: Package Restructuring

## What Changed

The package structure has been reorganized to support proper package imports:

**Before:**
```
src/
  helpers/
  inspect/
  data_generation/
```

**After:**
```
self_rec_framework/
  src/
    helpers/
    inspect/
    data_generation/
```

## Import Changes

### External Projects (Other Packages)

When installed as a package, use:

```python
from self_rec_framework.src.helpers.model_names import inspect_model_name
from self_rec_framework.src.helpers.model_sets import get_model_set
```

**Note:** Python package names cannot contain hyphens, so use `self_rec_framework` (underscores) in imports, even though the PyPI package name is `self-rec-framework` (hyphens).

### Internal Code (This Repository)

Internal code currently uses `from src.helpers import ...`. These imports need to be updated to:

```python
from self_rec_framework.src.helpers.model_names import inspect_model_name
```

Or, if running scripts from the repository root, you can add this to the top of scripts:

```python
import sys
from pathlib import Path
# Add parent directory to path for backward compatibility
sys.path.insert(0, str(Path(__file__).parent.parent))
```

## Migration Strategy

### Option 1: Update All Imports (Recommended)

Update all `from src.` imports to `from self_rec_framework.src.`:

```bash
# Find all files with src imports
find . -name "*.py" -type f -exec grep -l "from src\." {} \;

# Use sed or a script to update (be careful and test!)
# Example for a single file:
sed -i 's/from src\./from self_rec_framework.src./g' file.py
sed -i 's/import src\./import self_rec_framework.src./g' file.py
```

### Option 2: Add Compatibility Layer

Create a compatibility module that re-exports from the new location:

```python
# In self_rec_framework/src/__init__.py or a compatibility module
import sys
from pathlib import Path

# Add backward compatibility
import self_rec_framework.src.helpers as helpers_module
sys.modules['src.helpers'] = helpers_module
```

### Option 3: Use PYTHONPATH

When running scripts, add the parent directory to PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# or
PYTHONPATH="$(pwd):${PYTHONPATH}" python script.py
```

## Files That Need Updating

The following files import from `src` and need to be updated:

- `experiments/_scripts/utils.py`
- `experiments/_scripts/analysis/*.py` (multiple files)
- `experiments/_scripts/eval/*.py`
- `experiments/_scripts/gen/*.py`
- `scripts/verify_model_data.py`
- Files in `self_rec_framework/src/` (internal imports)

## Testing

After migration, verify imports work:

```python
# External import (after package installation)
from self_rec_framework.src.helpers.model_names import inspect_model_name
print(inspect_model_name("gpt-4o-mini"))

# Internal import (if updated)
from self_rec_framework.src.helpers.model_names import inspect_model_name
```

## Installation

Install the package in editable mode:

```bash
pip install -e .
```

This allows both:
- External projects to import: `from self_rec_framework.src.helpers import ...`
- Internal code to work (after import updates or compatibility layer)
