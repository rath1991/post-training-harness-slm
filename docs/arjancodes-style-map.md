# ArjanCodes: coding style and pattern mapping

This is a summary of the style taught by Arjan Egges (ArjanCodes), and the basis for the `functional-python` skill.

## Coding style
1. **Composition over inheritance.** Avoid deep class trees. Build behavior from small pieces.
2. **`Protocol` over `ABC`.** Use structural typing, so a class doesn't need to inherit anything to fit an interface. Use an ABC only when sharing real implementation.
3. **Functions first, classes when there is state.** A one-method class with no meaningful state becomes a function.
4. **Dataclasses everywhere.** Use `frozen=True`, `slots=True` and `field(default_factory=...)`. Keep data separate from behavior.
5. **Type hints throughout:** `Callable[...]`, type aliases, generics, and `ParamSpec` for decorators.
6. **Dependency injection.** Pass collaborators in, often as plain functions. Don't construct them inside.
7. **Small, single-purpose functions:** SOLID, GRASP (cohesion, low coupling), Law of Demeter.
8. **"Code Roast" refactors:** separate I/O from logic, remove globals, replace `if/elif` chains with dicts, move magic values into config.
9. **Pragmatism over purity.** No monads and no recursion-heavy code. Use functional tools where they make code simpler.

## Classic patterns and their functional versions
| Pattern | Classic OOP form | Pythonic, functional form |
|---|---|---|
| Strategy | Strategy ABC + subclasses | Pass a `Callable`; configure with `partial` or a closure |
| Observer / Event | Subject/Observer classes | `dict[str, list[Callable]]` of event name to handlers; `subscribe`/`post_event` functions |
| Factory | Factory class hierarchy | Dict from key to constructor or function |
| Command | Command class with `execute`/`undo` | Dataclass + `Protocol` for undoable commands; closures for simple ones |
| Template Method | Abstract base with hooks | A function taking the varying steps as callables |
| Decorator | Wrapper classes | Python decorators using `functools.wraps`, typed with `ParamSpec` |
| Chain of Responsibility / Pipeline | Linked handler objects | A list of functions, folded with `reduce` or a `compose` helper |
| Bridge | Two parallel hierarchies | `Protocol` on one side, injected callables on the other |
| Registry / Plugin | Metaclass registration | A dict filled by a `@register` decorator, or `importlib` loading from config |
| Singleton | Singleton class | A module-level instance, or better, construct once and inject |
| Builder | Builder class with chained methods | A dataclass with defaults, plus `dataclasses.replace` |

## Recurring toolkit
`functools.partial`, closures, dicts of functions for dispatch, `Callable` type aliases, `@cache`/`@lru_cache`, generators for lazy pipelines, `operator.itemgetter`/`attrgetter`, and a pure core with I/O at the edges.

The full catalog with code is in `claude-home/skills/functional-python/references/patterns.md`.
