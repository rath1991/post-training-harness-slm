# post-training-harness-slm

This repo holds my system-wide Claude Code setup for writing **clear, readable, mostly-functional Python**. It covers:

- **Core instructions** (`~/.claude/CLAUDE.md`): the coding philosophy, loaded in every session.
- **The `functional-python` skill**: ArjanCodes-style principles, a checklist, and a catalog of functional versions of classic design patterns.
- **The style gate** (`stylecheck.sh`): **isort, black, pycodestyle, pydocstyle** and **ruff** (functional-style rules). Code must pass all five.
- **A PostToolUse hook** that runs `stylecheck.sh --fix` after every `.py` edit Claude makes and sends any failures back to Claude.

## Install on a new machine

```bash
git clone https://github.com/rath1991/post-training-harness-slm.git
cd post-training-harness-slm
./install.sh            # macOS / Linux / WSL / Git Bash
```

Or open the repo in Claude Code and say **"install this harness system-wide"**. Claude follows `CLAUDE.md`, which also covers Windows.

## Use the style gate

```bash
stylecheck.sh path/to/file_or_dir      # check only; exit 0 = all pass
stylecheck.sh --fix path/              # apply isort + black first, then check
```

If a project has its own config for a tool (`pyproject.toml`, `setup.cfg`, `tox.ini`, ...), that config wins. Otherwise black-compatible defaults apply: line length 88, isort `--profile black`, Google docstrings.

## Layout

```
claude-home/                 # mirrors ~/.claude/
  CLAUDE.md                  # global instructions (merged, not overwritten)
  settings.hooks.json        # hook entry (merged into settings.json)
  skills/functional-python/  # SKILL.md + references/patterns.md
  hooks/python_style/        # stylecheck.sh, check.py, ruff.toml
docs/
  decisions.md               # why each choice was made
  arjancodes-style-map.md    # ArjanCodes style and pattern mapping
examples/                    # good.py / bad.py for verification
install.sh                   # idempotent installer
```
