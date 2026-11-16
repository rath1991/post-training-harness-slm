"""Order pricing: a small functional core used to verify the style gate."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from decimal import Decimal
from functools import partial

type Discount = Callable[[Decimal], Decimal]


@dataclass(frozen=True, slots=True)
class Item:
    """A line item in an order."""

    name: str
    price: Decimal
    quantity: int = 1


def no_discount(price: Decimal) -> Decimal:
    """Return the price unchanged."""
    return price


def percentage_discount(price: Decimal, percent: Decimal) -> Decimal:
    """Return the price reduced by the given percentage."""
    return price * (1 - percent / 100)


def subtotal(items: Iterable[Item]) -> Decimal:
    """Return the sum of price times quantity for all items."""
    return sum((item.price * item.quantity for item in items), Decimal(0))


def checkout(items: Iterable[Item], discount: Discount = no_discount) -> Decimal:
    """Return the total after applying the discount strategy."""
    return discount(subtotal(items))


def with_quantity(item: Item, quantity: int) -> Item:
    """Return a copy of the item with a new quantity."""
    return replace(item, quantity=quantity)


def main() -> None:
    """Print a sample checkout total."""
    cart = [Item("book", Decimal("12.50"), 2), Item("pen", Decimal("1.20"))]
    ten_percent_off = partial(percentage_discount, percent=Decimal(10))
    print(checkout(cart, ten_percent_off))


if __name__ == "__main__":
    main()
