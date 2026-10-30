import abc
from typing import (
    TYPE_CHECKING,
    Optional,
    TypeVar,
)

if TYPE_CHECKING:
    from .core import ClassNode


T = TypeVar("T")


class Descriptor(abc.ABC):
    """Abstract base class for descriptors."""

    @property
    @abc.abstractmethod
    def unique_id(self):  # pragma: no cover
        """Return a unique identifier for the descriptor."""
        ...

    @abc.abstractmethod
    def negate(self, negate: bool = True) -> "Descriptor":  # pragma: no cover
        """Negate the descriptor."""
        ...

    @abc.abstractmethod
    def format(self, anchor: Optional["ClassNode"] = None, quoted=False) -> str:
        """Format the descriptor as a string."""
        ...

    def __invert__(self):
        return self.negate()

    def __le__(self, other) -> bool:
        """Return True if self is implied by other."""
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if type(self) != type(other):
            return False

        return self.unique_id == other.unique_id  # type: ignore

    def __hash__(self) -> int:
        return hash(self.unique_id)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.format()}>"

    def __str__(self) -> str:
        return self.format()


class NeverDescriptor(Descriptor):
    """A descriptor that is never implied by another (even itself)."""

    @property
    def unique_id(self):
        raise NotImplementedError()

    def negate(self, negate: bool = True) -> Descriptor:
        raise NotImplementedError()

    def format(self, anchor: Optional["ClassNode"] = None) -> str:
        raise NotImplementedError()

    def __le__(self, other) -> bool:
        return False

    def __repr__(self) -> str:
        return object.__repr__(self)
