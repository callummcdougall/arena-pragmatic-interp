# FLAG_RUN Feature Documentation

## Overview

The `FLAG_RUN_*` feature allows you to conditionally run sections of code in master files during development, without affecting generated files (exercise notebooks, solution notebooks, streamlit pages, etc.).

## Purpose

This feature is designed for **development mode** to enable selective execution of code sections when testing material. For example, you might want to quickly test just Section 2 of your exercises without running the entire notebook.

## How It Works

### In Master Files (master.py and master.ipynb)

Flags are **preserved** in both master.py and master.ipynb files. You can:

1. **Define flags** anywhere in your code cells (typically in the imports cell):
   ```python
   FLAG_RUN_SECTION_1 = True
   FLAG_RUN_SECTION_2 = False
   FLAG_RUN_EXPENSIVE_COMPUTATION = False
   ```

2. **Use flags in `if MAIN:` blocks**:
   ```python
   # Simple flag usage
   if MAIN and FLAG_RUN_SECTION_1:
       train_model()
       evaluate_results()

   # Complex boolean expressions
   if MAIN and (FLAG_RUN_SECTION_1 or FLAG_RUN_SECTION_2):
       run_tests()

   # Multiple flags
   if MAIN and FLAG_RUN_SECTION_1 and FLAG_RUN_EXPENSIVE:
       expensive_computation()
   ```

### In Generated Files

Flags are **completely stripped** from all generated files. The conversion process:

1. **Removes flag definitions** entirely
2. **Replaces** `if MAIN and <flag_expression>:` with `if MAIN:`
3. **Processes normally** as if it were a standard `if MAIN:` block

This means:
- Exercise notebooks: Code runs as if there were no flags
- Solution notebooks: Code runs as if there were no flags
- Python solutions file: Has `if MAIN:` instead of `if MAIN and FLAG_...:`
- Streamlit files: Code runs as if there were no flags

## Usage Rules

### ✅ Valid Usage

```python
# Flag definitions (anywhere in code cells)
FLAG_RUN_SECTION_1 = True
FLAG_RUN_DEBUG = os.getenv("DEBUG", False)

# Simple flag condition
if MAIN and FLAG_RUN_SECTION_1:
    do_something()

# Complex boolean expressions
if MAIN and (FLAG_A or FLAG_B):
    do_something()

if MAIN and FLAG_A and FLAG_B and FLAG_C:
    do_something()

# Normal if MAIN without flags still works
if MAIN:
    always_runs()
```

### ❌ Invalid Usage

```python
# ERROR: FLAG must come after MAIN
if FLAG_RUN_SECTION_1 and MAIN:  # WRONG ORDER!
    do_something()

# ERROR: if MAIN and ... must contain FLAG
if MAIN and some_other_condition:  # MISSING FLAG!
    do_something()
```

## Conversion Flow

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│  master.py  │  ─────> │ master.ipynb │  ─────> │  master.py  │
│             │         │              │         │             │
│ WITH FLAGS  │         │ WITH FLAGS   │         │ WITH FLAGS  │
└─────────────┘         └──────────────┘         └─────────────┘
                                │
                                │ generate_files()
                                ▼
                        ┌───────────────┐
                        │ Generated     │
                        │ Files         │
                        │               │
                        │ WITHOUT FLAGS │
                        └───────────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
                    ▼           ▼           ▼
              ┌─────────┐ ┌─────────┐ ┌─────────┐
              │ Colab   │ │ Python  │ │Streamlit│
              │Exercise │ │Solutions│ │  Pages  │
              └─────────┘ └─────────┘ └─────────┘
```

## Example Workflow

### Step 1: Add flags to master.py

```python
# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

import torch
import numpy as np

# Development flags for selective execution
FLAG_RUN_SECTION_1 = True
FLAG_RUN_SECTION_2 = False
FLAG_RUN_SECTION_3 = True

# ... later in the file ...

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

if MAIN and FLAG_RUN_SECTION_1:
    # This section only runs if FLAG_RUN_SECTION_1 is True
    model = load_model()
    results = train_model(model)
```

### Step 2: Run conversion with `use_py=true`

```bash
cd infrastructure/core
python main.py --chapters=1.6.1 --use_py=true
```

The conversion process:
1. master.py → master.ipynb (flags preserved)
2. master.ipynb → master.py (flags preserved)
3. master.py → generated files (flags stripped)

### Step 3: Generated files have clean code

In the generated exercise notebook:
```python
# Flag definitions are gone
# if MAIN and FLAG_RUN_SECTION_1: becomes just the un-indented code
model = load_model()
results = train_model(model)
```

In the generated Python solutions file:
```python
# Flag definitions are gone
# if MAIN and FLAG_RUN_SECTION_1: becomes if MAIN:
if MAIN:
    model = load_model()
    results = train_model(model)
```

## Implementation Details

### Modified Files

1. **`processors.py`**:
   - Added `_strip_flags_from_source()` function to remove flag definitions and flag usage
   - Updated `_process_source()` to accept `strip_flags` parameter

2. **`cell.py`**:
   - Added validation in `_validate_and_fix_cell()` to check for invalid flag usage
   - Updated `process_code_cell()` to pass `strip_flags=True` when generating files
   - `master_ipynb_dict` property uses default `strip_flags=False` to preserve flags

### Validation

The system validates:
- ✓ `if MAIN and FLAG...` is valid
- ✗ `if FLAG... and MAIN` raises an error (wrong order)
- ✗ `if MAIN and ...` without FLAG raises an error (must contain FLAG if using "and")

## Benefits

1. **Development Speed**: Quickly test individual sections without running everything
2. **No Impact on Students**: Generated files are clean, students never see the flags
3. **Preserved Through Cycles**: Flags survive the full conversion cycle (py → ipynb → py)
4. **Flexible**: Supports any boolean expression with flags

## Notes

- Flags are **only** for master files during development
- Flag values (True/False) are preserved through conversion cycles
- You can have as many flags as you want
- Flag names must start with `FLAG_RUN_` (or just contain `FLAG`)
- Works seamlessly with existing `if MAIN:` blocks
