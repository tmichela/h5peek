# h5peek

A command-line tool for inspecting HDF5 files, written in Rust. It provides a colored tree view of the file structure, detailed datatype information, and interactive path completion.

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
```

## License

BSD-3-Clause
