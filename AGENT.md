# Daft Agent Guidelines

## Build/Test Commands
- `make build` - Build for development (uses maturin develop)
- `make test` - Run all tests (excludes integration tests)
- `pytest tests/path/to/test.py::test_function` - Run single test
- `pytest tests/path/to/test.py -k "test_pattern"` - Run tests matching pattern
- `make format` - Format Python (ruff) and Rust (rustfmt) code
- `make lint` - Lint Python and Rust code
- `make precommit` - Run all pre-commit hooks (format + lint)
- `cargo test -p daft-core` - Run Rust tests for specific crate

## Architecture
Daft is a distributed dataframe library with Rust core and Python bindings. Key components:
- **daft-core**: Core data types and functionality
- **daft-dsl**: Domain-specific language for expressions
- **daft-logical-plan**: Logical query planning layer
- **daft-physical-plan**: Physical execution planning
- **daft-ir**: Consolidated intermediate representation
- **daft-io**: I/O operations (S3, file systems)
- **daft-micropartition**: Partitioned data structures
- Built on Apache Arrow via arrow2 crate

## Code Style
- **Python**: Follow ruff config (.ruff.toml), line length 120, double quotes, Google docstrings
- **Rust**: Follow rustfmt.toml, group imports StdExternalCrate, use workspace lints
- **Imports**: Python requires `from __future__ import annotations` in main code
- **Testing**: Use pytest, exclude integration/benchmark tests by default
- **Error handling**: Use snafu for Rust errors, proper exception handling in Python
