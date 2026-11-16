# Pattern catalog: functional versions of classic patterns

Each entry shows the classic OOP shape, the version to write instead, and when the class version is still the right choice. Examples target Python 3.12+.

---

## Strategy -> pass a function

```python
# Avoid: an ABC plus one subclass per strategy
class DiscountStrategy(ABC):
    @abstractmethod
    def apply(self, price: Decimal) -> Decimal: ...

# Prefer
type Discount = Callable[[Decimal], Decimal]

def no_discount(price: Decimal) -> Decimal:
    return price

def percentage_discount(price: Decimal, percent: Decimal) -> Decimal:
    return price * (1 - percent / 100)

def checkout(items: Sequence[Item], discount: Discount = no_discount) -> Decimal:
    return discount(sum((item.price for item in items), Decimal(0)))

checkout(cart, partial(percentage_discount, percent=Decimal(10)))
```
Use a class (via a Protocol) instead when a strategy needs several coordinated methods.

---

## Factory -> dict of constructors

```python
type Exporter = Callable[[Video, Path], None]

EXPORTERS: Mapping[str, Exporter] = {
    "mp4": export_mp4,
    "webm": export_webm,
}

def get_exporter(fmt: str) -> Exporter:
    try:
        return EXPORTERS[fmt]
    except KeyError:
        raise ValueError(f"Unknown format {fmt!r}; choose from {sorted(EXPORTERS)}") from None
```

---

## Registry / plugin -> decorator that fills a dict

```python
type Handler = Callable[[Event], None]
HANDLERS: dict[str, Handler] = {}

def handles(event_type: str) -> Callable[[Handler], Handler]:
    def register(handler: Handler) -> Handler:
        HANDLERS[event_type] = handler
        return handler
    return register

@handles("user_signed_up")
def send_welcome_email(event: Event) -> None: ...
```
The registry is module-level mutable state that is written only at import time. That's acceptable; mutating it at runtime is not.

---

## Observer / events -> subscribers as functions

```python
type Listener = Callable[[Event], None]

def subscribe(listeners: Mapping[str, tuple[Listener, ...]], name: str, listener: Listener) -> dict[str, tuple[Listener, ...]]:
    return {**listeners, name: (*listeners.get(name, ()), listener)}

def publish(listeners: Mapping[str, Sequence[Listener]], event: Event) -> None:
    for listener in listeners.get(event.name, ()):
        listener(event)
```
A small `EventBus` class is fine when subscriptions change at runtime throughout the program's life. Give it an explicit owner and inject it rather than making it global.

---

## Template method -> higher-order function

```python
def run_pipeline(
    load: Callable[[], Iterable[Row]],
    transform: Callable[[Row], Record],
    save: Callable[[Iterable[Record]], None],
) -> None:
    save(transform(row) for row in load())
```

---

## Command (with undo) -> frozen dataclass + Protocol

```python
class Command(Protocol):
    def execute(self, state: Doc) -> Doc: ...
    def undo(self, state: Doc) -> Doc: ...

@dataclass(frozen=True, slots=True)
class InsertText:
    position: int
    text: str

    def execute(self, state: Doc) -> Doc:
        return replace(state, body=state.body[:self.position] + self.text + state.body[self.position:])

    def undo(self, state: Doc) -> Doc:
        return replace(state, body=state.body[:self.position] + state.body[self.position + len(self.text):])
```
The commands are values and the state is immutable, so the history is a plain `list[Command]`. Commands without undo can simply be functions.

---

## Decorator -> typed function decorator

```python
def retry[**P, R](attempts: int) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except TransientError:
                    if attempt == attempts:
                        raise
            raise AssertionError("unreachable")
        return wrapper
    return decorate
```

---

## Chain of responsibility / pipeline -> list of functions

```python
type Step = Callable[[Text], Text]

def compose_steps(steps: Iterable[Step]) -> Step:
    return lambda text: reduce(lambda acc, step: step(acc), steps, text)

clean = compose_steps([strip_html, normalize_whitespace, lowercase])
```
Keep the pipelines short and use named steps. If a reader has to trace the composition to understand it, write the calls out explicitly instead.

---

## Builder -> dataclass with defaults

```python
@dataclass(frozen=True, slots=True)
class Request:
    url: str
    method: str = "GET"
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout: float = 10.0

req = replace(Request("https://..."), method="POST")
```

---

## Singleton -> module-level instance or injection

Construct the object once in the shell (`main`) and pass it down. If a module-level instance is needed, make it read-only configuration.

---

## Dependency injection -> arguments (and partial at the edge)

```python
def sync_users(fetch_users: Callable[[], Iterable[User]], save_user: Callable[[User], None]) -> int:
    count = 0
    for user in fetch_users():
        save_user(user)
        count += 1
    return count

# shell
def main() -> None:
    client = ApiClient(os.environ["API_URL"])
    db = Database.connect(os.environ["DB_URL"])
    synced = sync_users(client.list_users, db.save_user)
    print(f"Synced {synced} users")
```
In tests, pass lists and lambdas. No mocks needed.

---

## Functional core, imperative shell: module layout

```python
"""Invoice report."""

# 1. imports
# 2. types and data (frozen dataclasses, Enums, type aliases)
# 3. pure functions (parsing, rules, calculations, formatting)
# 4. shell: the only place that touches files, network, env, clock, stdout

def main() -> None:
    raw = Path(sys.argv[1]).read_text()
    invoices = parse_invoices(raw)                       # pure
    report = format_report(summarize(invoices, today=date.today()))  # pure
    print(report)

if __name__ == "__main__":
    main()
```

---

## Dispatch: dict vs match

- Use a **dict of functions** when keys are simple values and the set may grow, such as commands, formats, or event names.
- Use **`match`** when dispatching on shape or type: `case Circle(radius=r)`, `case {"type": "error", "msg": msg}`.
- Use **`functools.singledispatch`** when the behavior extends by type across modules.
