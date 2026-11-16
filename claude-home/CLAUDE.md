# Global instructions

## Python
- Clear, readable code is the goal. My coding philosophy is **mostly functional programming**: pure functions, immutable data (frozen dataclasses), composition, functions passed as values, and side effects at the edges ("functional core, imperative shell"). Objects and methods are fine when they make code clearer.
- Style reference: ArjanCodes. Prefer `Protocol` over ABCs, composition over inheritance, dict dispatch over `if/elif` chains, `partial`/closures over one-method classes, and dependency injection over hidden construction.
- Load the `functional-python` skill before writing, editing, reviewing or designing any Python.
- Type-hint every function signature. Use no boolean flag parameters, mutable defaults, or global mutable state.
- In existing projects, follow the established conventions, and don't rewrite code outside the task's scope.
- Code style checks must pass: **black, isort, pycodestyle, pydocstyle** (plus ruff for functional-style rules), all run by `~/.claude/hooks/python_style/stylecheck.sh` (on PATH as `stylecheck.sh`). A hook runs `stylecheck.sh --fix` after every `.py` edit. Before calling Python work done, run `stylecheck.sh <changed paths>` and confirm it exits 0.
