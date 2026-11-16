# Decisions and rationale

This file records how the harness was designed and why. It lets Claude (or me) on another machine reconstruct the intent, not just the files.

## Goal
Write **clear, readable Python**. My coding philosophy is **mostly functional programming**. Objects and methods are allowed when they make code clearer. Readability is the goal and FP is the main tool for getting there.

## 1. Style reference: ArjanCodes
The style follows Arjan Egges (ArjanCodes): composition over inheritance, `Protocol` over `ABC`, functions as values (`Callable`, `partial`, closures), dict dispatch, frozen dataclasses, dependency injection, "functional core, imperative shell", and type hints everywhere. It is pragmatic rather than purist: no monad libraries, and no recursion for plain iteration. See `arjancodes-style-map.md`.

## 2. Three layers of enforcement
| Layer | File | Why |
|---|---|---|
| Core instructions | `~/.claude/CLAUDE.md` | Always in context. Short enough to read every session. Points to the skill. |
| Skill | `~/.claude/skills/functional-python/` | The full principles, a checklist, and the pattern catalog. Loaded on demand for any Python work so it doesn't bloat every session. |
| Hook + gate | `~/.claude/hooks/python_style/` | Deterministic enforcement that doesn't depend on the model remembering the rules. |

## 3. The style gate: one shell script
Required: **black, isort, pycodestyle, pydocstyle**, all run from one executable shell script that must pass (exit 0). **ruff** was added as a fifth check for the functional and readability rules that the other four don't cover: mutable defaults, `global`, boolean positional flags, missing annotations, too many arguments or branches, and complexity.

- **Black-compatible defaults:** line length 88, isort `--profile black`, and pycodestyle ignoring E203, E701, E704 and W503 (rules that conflict with black's output).
- **pydocstyle uses the Google convention** and ignores D105 (magic methods) and D107 (`__init__`). `test_*.py` files are exempt by pydocstyle's default.
- **Project config wins.** If a repo configures a tool itself, the gate doesn't override it. This respects existing projects.
- **Thresholds** (`ruff.toml`): max 5 arguments, complexity 10, 10 branches, 40 statements.

## 4. Hook behavior
- A PostToolUse hook on `Edit|Write|MultiEdit` runs only for `.py`/`.pyi` files.
- It runs `stylecheck.sh --fix`. isort and black are applied automatically, because they are deterministic and requiring them to pass implies it. Then every check runs.
- Exit 2 sends the failures to Claude as feedback. Any other failure, such as missing tools or a crash, stays silent and **never blocks work**.
- **Trade-off accepted:** in repos that don't use black, the first edit reformats the whole file, which produces a large diff.
- **Opt-out:** a file whose first line is `# stylecheck: skip` is skipped by the hook. This is for intentional fixtures like `examples/bad.py`. A direct `stylecheck.sh` run still checks it.
- The hook calls `bash stylecheck.sh` explicitly so it works on Windows (Git Bash) too.

## 5. Tools
These are installed user-wide with `uv tool install` (fallbacks: pipx, then pip `--user`). `stylecheck.sh` is symlinked into `~/.local/bin`.

## Possible future changes
- Replace pydocstyle (no longer maintained) with ruff's `D` rules to cut down the number of tools.
- Add `mypy` or `pyright` to the gate for type checking.
