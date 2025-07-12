# AGENTS.md

## Build, Lint, and Test Commands

- **Install dependencies:** `pip install -r requirements.txt`
- **Run linter:** `flake8 .`
- **Run all tests:** No tests found (add tests in the future).
- **Run a single test:** `pytest path/to/test_file.py::test_function` (when tests exist).

## Code Style Guidelines

- **Formatting:** Follow PEP 8 with 4-space indentation.
- **Imports:** Group imports (standard library, third-party, local) and sort alphabetically.
- **Naming:**
    - `PascalCase` for classes
    - `snake_case` for functions and variables
    - `UPPER_SNAKE_CASE` for constants
- **Types:** Type hints are encouraged for new code, but not required.
- **Error Handling:** Use `try...except` for operations that might fail; provide fallback mechanisms.
- **Docstrings:** Add docstrings to all public functions and classes to explain their purpose.

## Agentic Best Practices

- Follow these guidelines for all code contributions.
- If adding tests, ensure they are discoverable by pytest.
- No Cursor or Copilot rules are present in this repository.
