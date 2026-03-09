# h5peek

A command-line tool for inspecting HDF5 files, written in Rust. It provides a colored tree view of the file structure, detailed datatype information, and interactive path completion.
This started as a re-implementation of the Python [h5glance](https://pypi.org/project/h5glance/) application, but might derive from it in future.

## Features

*   **Tree View**: Visualize groups, datasets, and links hierarchy.
*   **Rich Details**: Displays shapes, and detailed types (e.g., `64-bit floating point`, `compound (x: int32, y: float64)`).
*   **Colors**: Semantic coloring for groups, datasets, and links (respects `NO_COLOR`).
*   **Interactive Mode**: Tab-completion for exploring paths deep inside files.
*   **Link Handling**: Correctly identifies and displays Soft and External links.
*   **Attribute Inspection**: View attribute names, types, and shapes.

## Installation

### From Source

```bash
cargo install --path .
```

## Usage

```bash
# View file structure
h5peek file.h5

# View specific group or dataset
h5peek file.h5 /group/dataset

# Interactive mode with tab completion
h5peek file.h5 -

# Show attributes
h5peek file.h5 --attrs

# Limit depth
h5peek file.h5 --depth 2

# Preserve original (unsorted) member order
h5peek file.h5 --unsorted
```

## Color output

By default, `h5peek` uses color when stdout is a terminal and `NO_COLOR` is not set.

```bash
# Force color output (even when piped)
h5peek file.h5 --color always

# Disable color output
h5peek file.h5 --color never
```

When using a pager (`--pager`), `h5peek` will force color unless `NO_COLOR` is set or `--color never` is used.

## Error handling

- Metadata always prints for datasets; data previews are best-effort.
- When `--slice` is provided, slice parsing or read failures exit non-zero and report an error.
- When `--slice` is not provided, preview failures are printed as warnings and the command continues.


## Sample File

```bash
# Install deps for the generator
python3 -m pip install h5py numpy

# Generate a sample file for testing
python3 script/generate_sample.py data/sample.h5
```

## License

MIT
