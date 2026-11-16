---
name: functional-python
description: The user's Python coding philosophy - clear, readable, mostly-functional Python in the style of ArjanCodes (pure functions, frozen dataclasses, Protocols, composition, functions passed as values, side effects at the edges). TRIGGER whenever writing, editing, refactoring, reviewing or designing Python code of any size, including scripts, notebooks, tests and one-off snippets. SKIP only for non-Python languages.
---

# Functional Python

Readable code is the goal; functional programming is the main way to get there. Objects and methods are fine where they make code clearer. The deciding question is always: **will the next reader understand this faster?**

The pattern catalog with before/after code is in `references/patterns.md`. Read it when designing a module, choosing a pattern, or refactoring.

## Core principles

### 1. Data and behavior stay separate
- Model data as `@dataclass(frozen=True, slots=True)`, `NamedTuple`, `Enum`, or plain tuples/dicts at boundaries.
- Don't mutate data you were given. Return a new value instead: `dataclasses.replace(order, status=Status.PAID)`, `{**config, "debug": True}`, `[*items, new_item]`.
- Behavior lives in module-level functions that take data and return data.
- Use `field(default_factory=list)`, never a mutable default.

### 2. Pure functions by default
- A function's output depends only on its inputs. It doesn't read globals, clocks, environment variables, or the network.
- Inject impure things as arguments: `now: datetime`, `fetch: Callable[[str], bytes]`, `rng: random.Random`.
- **Functional core, imperative shell.** Keep I/O (files, DB, HTTP, printing, logging config, `sys.argv`) in a thin outer layer, usually `main()` or entry points. Everything it calls into is pure and easy to test.
- Mutation local to a function is fine when it can't be seen from outside, for example building a list in a loop that would be an unreadable comprehension.

### 3. Functions are values
- Pass behavior in as `Callable` arguments instead of building class hierarchies (Strategy, Template Method, Observer).
- Name the signature with a type alias: `type Discount = Callable[[Decimal], Decimal]` (3.12+) or `Discount = Callable[...]`.
- Configure functions with `functools.partial` or closures instead of one-method classes.
- Replace `if/elif` chains over keys with a dict of functions. Use `match` for structural cases.

### 4. Composition over inheritance
- No inheritance for code reuse. Compose small functions and objects.
- Define interfaces with `typing.Protocol` (structural), not `ABC`. Use an ABC only when sharing real implementation, and keep it one level deep.
- Inject dependencies. Don't construct them inside the function that uses them.

### 5. When a class is the right tool
Use a class when at least one of these holds:
- State with invariants that must change together, such as a cache, connection pool, or game state.
- A resource lifecycle (`__enter__`/`__exit__`).
- A framework requires it (Django models, PyTorch `nn.Module`, pytest fixtures returning objects).
- Several functions share the same 3+ dependencies and passing them around hurts readability.

A class with one method and `__init__` should be a function or a `partial`. Keep methods short, and let them delegate to pure functions where possible.

### 6. Small, honest functions
- A function does one thing at one level of abstraction. Around 20 lines is a smell, not a rule.
- Use 4 parameters or fewer. Beyond that, group them into a dataclass.
- **No boolean flag parameters.** Split into two functions, use an `Enum`, or make the flag keyword-only (`*, dry_run: bool = False`) when it truly toggles one detail.
- Return early instead of nesting. Maximum nesting depth is about 2.
- A name says what the function returns or does: `parse_invoice`, `total_price`, `is_overdue`. Data gets a noun. No abbreviations.

### 7. Types everywhere
- Annotate all function signatures with modern syntax: `list[int]`, `dict[str, float]`, `X | None`, `Self`, `type` aliases, `ParamSpec` for decorators.
- Prefer abstract parameter types (`Iterable`, `Sequence`, `Mapping`) and concrete return types.
- `Any` is a last resort. `cast` needs a comment explaining why.

### 8. Iteration, readably
- Comprehensions beat `map`/`filter` with a lambda. `map(parse_row, rows)` with a named function is fine.
- Use generators and `itertools` for lazy pipelines over large or streaming data.
- Use `sum`, `any`, `all`, `max(key=...)`, `sorted(key=...)` before `functools.reduce`. Use `reduce` only when the fold is obvious.
- `operator.itemgetter`/`attrgetter` replace trivial lambdas.
- Don't use recursion for plain iteration (Python has no tail-call optimization). Recursion is fine for recursive data such as trees.

### 9. Errors
- Validate at the boundary (parse, don't validate: convert raw input into typed dataclasses once, at the edge).
- Raise specific exceptions, and define a small domain exception if needed. Don't return `None` or sentinel values to signal failure unless absence really is a normal result.
- Don't introduce Result/Maybe monad libraries unless the project already uses them.

### 10. Avoid
Global mutable state, `global`, singletons, metaclasses, deep inheritance, mutable default arguments, clever one-liners, long `compose(...)` point-free chains, lambdas longer than one short expression, `*args/**kwargs` pass-through that hides the real signature, and comments that restate the code. Comments should say *why*.

## Working in existing code
- Follow the project's established conventions (formatter, typing level, framework idioms) where they conflict with this skill, and say so in one line.
- Apply this style to code you write or touch. Don't rewrite unrelated code unless asked. Do mention a refactoring opportunity if it's significant.
- Pick the smallest change that makes touched code clearer.

## Before you finish: checklist
1. Can each function be tested without mocks? If not, is the I/O at the edge?
2. Is any input mutated? Is there any global or hidden state?
3. Is there a class that should be a function, or a hierarchy that should be a `Callable`/`Protocol`?
4. Are there boolean flags, more than 4 params, nesting deeper than 2, or long `if/elif` dispatch?
5. Are all signatures typed, with names that read like prose?
6. Would a reader new to the module understand it top to bottom? Order the module as: imports, types/data, pure functions, shell/entry point.

## Style gate (must pass)
All Python must pass `stylecheck.sh` (`~/.claude/hooks/python_style/stylecheck.sh`, also on PATH):

| Tool | Enforces | Default config (used when the project has none) |
|---|---|---|
| isort | import order | `--profile black` |
| black | formatting | line length 88 |
| pycodestyle | PEP 8 | max line 88; ignores E203, E701, E704, W503 (black-compatible) |
| pydocstyle | docstrings | Google convention. Every public module, class and function needs a docstring (`test_*.py` files are exempt). |
| ruff | functional-style rules | `hooks/python_style/ruff.toml` |

- Write code that passes from the start. Use Google-style docstrings with a one-line summary in the imperative mood, and add `Args:`/`Returns:`/`Raises:` when they aren't obvious from the signature. Keep lines to 88 characters or fewer.
- The PostToolUse hook runs `stylecheck.sh --fix <file>` after every edit. It auto-applies isort and black, so **re-read the file before editing it again**. Any remaining failures come back to you. Fix them in code you touched.
- A file whose first line is `# stylecheck: skip` is ignored by the hook, though a direct `stylecheck.sh` run still checks it. Use this only for deliberate fixtures such as intentionally broken test inputs, and never to dodge findings.
- Before declaring Python work done, run `stylecheck.sh <changed paths>` and confirm it exits 0. Report the result.
