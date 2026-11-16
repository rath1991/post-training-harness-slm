# Instructions for Claude: install this harness system-wide

This repo holds the user's **system-wide Claude Code setup for Python**. It isn't an application. When the user asks you to "set up", "install", "apply" or "sync" this harness, or opens this repo and asks what to do, carry out the steps below on **this machine**.

## What gets installed into `~/.claude/`

| Source in repo | Destination | Purpose |
|---|---|---|
| `claude-home/CLAUDE.md` | merged into `~/.claude/CLAUDE.md` | global core instructions (functional Python philosophy and style gate) |
| `claude-home/skills/functional-python/` | `~/.claude/skills/functional-python/` | the skill: principles, checklist, pattern catalog |
| `claude-home/hooks/python_style/` | `~/.claude/hooks/python_style/` | `stylecheck.sh` (style gate), `check.py` (hook), `ruff.toml` |
| `claude-home/settings.hooks.json` | merged into `~/.claude/settings.json` | PostToolUse hook on `Edit\|Write\|MultiEdit` |
| tools | user-level install | `black`, `isort`, `pycodestyle`, `pydocstyle`, `ruff` |

Read `docs/decisions.md` for *why* each piece exists, and `docs/arjancodes-style-map.md` for the style background.

## Steps

1. **Detect the platform**: macOS/Linux, Windows with Git Bash, or WSL. Find which Python command works (`python3`, `python` or `py -3`) and whether `uv` is installed.
2. **Inspect before overwriting.** Read the existing `~/.claude/CLAUDE.md`, `~/.claude/settings.json`, and any existing `~/.claude/skills/functional-python` or `~/.claude/hooks/python_style`. If they hold anything not from this repo, tell the user and merge it. Never discard their content.
3. **macOS / Linux / WSL / Git Bash:** run `./install.sh`. It's idempotent: it backs up the old files to `~/.claude/backups/harness-<timestamp>/`, copies the files, merges settings and CLAUDE.md, installs the tools and runs the verification.
4. **Windows without bash in PATH:** do the same steps by hand:
   - Copy the files as in the table.
   - Merge `claude-home/settings.hooks.json` into `settings.json` and **change the hook command's interpreter** from `python3` to whatever works on this machine (e.g. `python` or `py -3`). Use a path with forward slashes, or `%USERPROFILE%`.
   - Install the tools: `uv tool install <tool>` for each one, or `pipx install`, or `python -m pip install --user`.
   - `check.py` calls `bash stylecheck.sh`, so Git Bash must be installed (Claude Code on Windows needs it anyway). If `bash` isn't on PATH, add Git's `bin` directory or point to `C:/Program Files/Git/bin/bash.exe`.
5. **Verify.** All of these must hold:
   - `stylecheck.sh examples/good.py` exits **0**.
   - `stylecheck.sh examples/bad.py` exits **1** and reports failures from all five tools.
   - Piping `{"tool_input":{"file_path":"<abs path>/examples/bad.py"}}` into `check.py` exits **2**. Run this on a **copy** of `bad.py`, because the hook runs `--fix` and reformats the file.
   - `~/.claude/settings.json` is still valid JSON, contains exactly one `python_style/check.py` hook, and keeps all of the user's other settings.
6. **Report** what was installed, what was merged or changed, the location of the backups, and the verification results. The skill and CLAUDE.md take effect in the next Claude Code session. The hook is live straight away.

## Updating the harness

If the user edits the live files in `~/.claude/` and wants them back in this repo, copy them into `claude-home/`. Then run `stylecheck.sh claude-home/hooks/python_style/check.py` and commit on a branch.

## Rules for this repo
- Python in this repo follows the harness itself, so `stylecheck.sh` must pass.
- Keep `install.sh` idempotent and non-destructive: always back up, always merge, never overwrite blindly.
