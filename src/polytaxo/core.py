import functools
import itertools
import operator
from textwrap import dedent, indent
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from .alias import Alias
from .descriptor import Descriptor, NeverDescriptor
from .parser import quote, tokenize

F = TypeVar("F", bound=Callable[..., Any])


def fill_in_doc(fields: Mapping[str, str]) -> Callable[[F], F]:
    fields = {k: dedent(v) for k, v in fields.items()}

    def decorator(decorated: F) -> F:
        if decorated.__doc__:
            decorated.__doc__ = dedent(decorated.__doc__).format_map(fields)

        return decorated

    return decorator


_doc_fields = {
    "anchor_arg": """anchor (ClassNode): The starting point for interpreting descriptors.""",
    "on_conflict_arg": """on_conflict ("replace", "raise", or "skip", optional): Conflict resolution strategy. Defaults to "replace".""",
    "ignore_unmatched_intermediaries_arg": """ignore_unmatched_intermediaries (bool, optional): Whether to ignore unmatched intermediaries. Defaults to False.""",
    "with_alias_arg": """with_alias (bool, optional): Whether to consider aliases. Defaults to False.""",
}


class NodeNotFoundError(Exception):
    """Exception raised when a node is not found."""

    pass


class CoreDescriptor(Descriptor):
    """A common base class that makes Descriptors comparable to Descriptions."""

    def __le__(self, other) -> bool:
        if isinstance(other, Description):
            for other_descriptor in other.descriptors:
                if self <= other_descriptor:
                    return True
            return False

        return NotImplemented  # pragma: no cover


def _validate_name_or_path(name_or_path: Union[str, Sequence[str]]) -> Sequence[str]:
    if isinstance(name_or_path, str):
        return (name_or_path,)
    return name_or_path


class BaseNode:
    """Base class for all taxonomy nodes."""

    def __init__(
        self,
        name: str,
        parent: Optional["BaseNode"],
        aliases: Optional[Iterable[str]] = None,
    ) -> None:
        self.name = name
        self.parent = parent

        if aliases is None:
            aliases = tuple()
        elif isinstance(aliases, str):
            aliases = (aliases,)

        self.aliases = tuple(Alias(a) for a in aliases)

    @functools.cached_property
    def precursors(self) -> Tuple["BaseNode", ...]:
        """Return a tuple of all precursor nodes up to the root."""

        def _(node):
            while node is not None:
                yield node
                node = node.parent

        return tuple(_(self))[::-1]

    @property
    def siblings(self):
        """Return a tuple of sibling nodes (with the same type as self)."""
        if self.parent is None:
            return tuple()

        if isinstance(self, ClassNode) and isinstance(self.parent, ClassNode):
            return tuple(s for s in self.parent.classes if s is not self)
        elif isinstance(self, TagNode) and isinstance(self.parent, TagNode):
            return tuple(s for s in self.parent.tags if s is not self)

        return tuple()

    @functools.cached_property
    def path(self):
        """Return the path from the root to this node as a tuple of names."""
        return tuple(n.name for n in self.precursors)

    def format(self, anchor: Union["ClassNode", None] = None, quoted=False) -> str:
        """Format the node as a string relative to an optional anchor."""
        precursors = self.precursors

        if anchor is not None:
            # TagNodes are displayed relative to the next precursor of anchor
            if isinstance(self, TagNode):
                # Find first ClassNode in precursors
                tag_anchor = self.parent_class
                if tag_anchor in anchor.precursors:
                    anchor = tag_anchor

            if anchor == self:
                return self.name

            i = precursors.index(anchor)
            precursors = precursors[i + 1 :]

        def build():
            sep = ""
            for n in precursors:
                yield sep + n.name
                if isinstance(n, TagNode):
                    sep = ":"
                else:
                    sep = "/"

        result = "".join(build())
        if quoted:
            return quote(result)
        return result

    def _matches_name(self, name: str, with_alias: bool) -> int:
        """Check if the node matches the given name or alias."""
        if name.casefold() == self.name.casefold():
            return 1 + len(name)

        if with_alias:
            return max((a.match(name) for a in self.aliases), default=0)

        return 0


TBaseNode = TypeVar("TBaseNode", bound=BaseNode)


def _best_match(
    anchor: BaseNode, name_or_path, matches: Iterable[Tuple[TBaseNode, int]]
) -> TBaseNode:
    matches = list(matches)

    if not matches:
        raise NodeNotFoundError(
            f"Could not find {name_or_path} from anchor {quote(anchor.format())}"
        )

    # Select most specific match
    matches.sort(key=operator.itemgetter(1), reverse=True)
    return matches[0][0]


class IndexProvider:
    """Class for providing unique indices for nodes."""

    def __init__(self) -> None:
        self.counter = itertools.count()
        self.taken: Set[int] = set()

    def remove(self, index):
        """Remove a specific index from the available pool."""
        if index in self.taken:
            raise ValueError(f"Index {index} is already taken")

        self.taken.add(index)

    def take(self):
        """Take the next available index."""
        while True:
            index = next(self.counter)
            if index not in self.taken:
                break

        self.taken.add(index)

        return index

    @property
    def n_labels(self):
        """Return the number of unique labels."""
        return max(self.taken) + 1


class RealNode(BaseNode, CoreDescriptor):
    """Base class for real nodes (ClassNode, TagNode)."""

    def __init__(
        self,
        name: str,
        parent: Optional["BaseNode"],
        index: Optional[int],
        meta: Optional[Mapping],
        aliases: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(name, parent, aliases)
        self.index = index

        if meta is None:
            meta = {}

        self.meta = meta

    def to_dict(self) -> Dict[str, Any]:
        """Convert the node to a dictionary representation."""
        d: Dict[str, Any] = {}
        if self.index is not None:
            d["index"] = self.index
        return d

    @property
    def unique_id(self):
        return self.path

    @property
    def real_children(self) -> List["RealNode"]:
        raise NotImplementedError()

    def walk(self) -> Iterator["RealNode"]:
        """Walk through the node and its children."""
        yield self
        for child in self.real_children:
            yield from child.walk()

    def negate(self, negate: bool = True) -> Descriptor:
        if negate:
            return NegatedRealNode(self)
        return self


TRealNode = TypeVar("TRealNode", bound=RealNode)


class NegatedRealNode(CoreDescriptor, Generic[TRealNode]):
    """Represents a negated real node."""

    is_negated: bool = True

    def __init__(self, node: TRealNode) -> None:
        # We cannot negate the root
        if node.parent is None:
            raise ValueError(f"Root node cannot be negated")

        self.node = node

    def format(self, anchor: Union["ClassNode", None] = None, quoted=False) -> str:
        """Format the negated node as a string."""
        return f"!{self.node.format(anchor, quoted=quoted)}"

    @property
    def unique_id(self):
        return ("!",) + self.node.path

    def negate(self, negate: bool = True) -> Descriptor:
        if negate:
            return self.node
        return self

    def __le__(self, other) -> bool:
        if isinstance(other, NegatedRealNode):
            # !A <= !B iff B <= A
            return other.node <= self.node

        if isinstance(self.node, ClassNode):
            if isinstance(other, TagNode):
                return False

            if isinstance(other, ClassNode):
                # !A <= B if A is a sibling of any precursor of B
                for n in other.precursors[::-1]:
                    if any(s <= self.node for s in n.siblings):
                        return True

                return False

        if isinstance(self.node, TagNode):
            if isinstance(other, ClassNode):
                return False

            if isinstance(other, TagNode):
                # !A <= B iff A implies a sibling of any precursor of B
                for n in other.precursors[::-1]:
                    if not isinstance(n, TagNode):
                        break

                    if any(s <= self.node for s in n.siblings):
                        return True

                return False

        return super().__le__(other)


class TagNode(RealNode):
    """Represents a tag node."""

    parent: Union["ClassNode", "TagNode"]

    def __init__(
        self,
        name: str,
        parent: Union["ClassNode", "TagNode"],
        index: Optional[int],
        meta: Optional[Mapping] = None,
        aliases: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(name, parent, index, meta, aliases)

        self.tags: List["TagNode"] = []

    @staticmethod
    def from_dict(
        name, data: Optional[Mapping], parent: Union["ClassNode", "TagNode"]
    ) -> "TagNode":
        """Create a TagNode from a dictionary representation."""
        if data is None:
            data = {}

        tag_node = TagNode(
            name,
            parent,
            data.get("index"),
            data.get("meta"),
            data.get("alias"),
        )

        for tag_name, tag_data in data.get("tags", {}).items():
            tag_node.add_tag(TagNode.from_dict(tag_name, tag_data, tag_node))

        return tag_node

    def to_dict(self) -> Dict[str, Any]:
        """Convert the TagNode to a dictionary representation."""
        d = super().to_dict()
        if self.tags:
            d["tags"] = {c.name: c.to_dict() for c in self.tags}

        return d

    def add_tag(self, tag: "TagNode"):
        """Add a child TagNode."""
        self.tags.append(tag)
        return tag

    @property
    def real_children(self):
        return self.tags

    @functools.cached_property
    def parent_class(self) -> "ClassNode":
        """Return the class parent (first parent that is a ClassNode) of the TagNode."""
        parent = self.parent
        while parent is not None:
            if isinstance(parent, ClassNode):
                return parent
            parent = parent.parent

        raise ValueError(f"Tag without a class parent: {self.name}")  # pragma: no cover

    def _find_all_tag(
        self, path: Sequence[str], with_alias=False, _base_specificy=0
    ) -> Iterable[Tuple["TagNode", int]]:
        """Find all TagNode instances matching the given path."""

        name, *tail = path

        specificy = self._matches_name(name, with_alias)

        # If self matches, descend with rest of the path
        if specificy:
            if tail:
                for subtag in self.tags:
                    yield from subtag._find_all_tag(
                        tail, with_alias, specificy + _base_specificy
                    )
            else:
                yield (self, specificy + _base_specificy)

        # Also descend with the full path to find nodes that didn't specify the full path
        for subtag in self.tags:
            yield from subtag._find_all_tag(path, with_alias, _base_specificy)

    def find_tag(
        self, name_or_path: Union[str, Sequence[str]], with_alias=False
    ) -> "TagNode":
        """Find a tag by name or path."""
        name_or_path = _validate_name_or_path(name_or_path)

        return _best_match(
            self, name_or_path, self._find_all_tag(name_or_path, with_alias)
        )

    def format_tree(self, extra_info=None) -> str:
        """Format the TagNode and its children as a tree."""
        name = self.name

        attrs = []
        if self.index is not None:
            attrs.append(f"index={self.index}")

        if self.meta:
            attrs.append(f"meta={self.meta}")

        if attrs:
            name = name + " (" + (", ".join(attrs)) + ")"

        if (self.index is not None) and (extra_info is not None):
            name += f" [{extra_info[self.index]}]"

        lines = [f":{name}"]

        for subtag in self.tags:
            lines.append(indent(subtag.format_tree(extra_info), "  "))

        return "\n".join(lines)

    @functools.cached_property
    def rivalling_children(self):
        """The set of directly rivalling descendants."""

        result: List[TagNode] = []
        for tag in self.tags:
            if tag.index is not None:
                result.append(tag)
            else:
                result.extend(tag.rivalling_children)

        return result

    def __le__(self, other) -> bool:
        if isinstance(other, TagNode):
            return self in other.precursors

        if isinstance(other, ClassNode):
            return False

        if isinstance(other, NegatedRealNode):
            return False

        return super().__le__(other)


class VirtualNode(BaseNode):
    """
    Represents a virtual node.

    Virtual nodes are used in `Description.from_lineage` to map
    external compound concepts to individual PolyTaxo concepts.
    """

    def __init__(
        self,
        name: str,
        parent: Optional["BaseNode"],
        description: "Description",
    ) -> None:
        super().__init__(name, parent)

        self.description = description

    def format_tree(self) -> str:
        """Format the virtual node as a tree."""
        return f":{self.name} -> {self.description!s}"

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.format_tree()}>"


class ClassNode(RealNode):
    """
    Represents a class.

    Classes form the backbone of the PolyTaxo tree.
    They can have subclasses, tags, and virtual nodes.
    """

    parent: Optional["ClassNode"]

    def __init__(
        self,
        name: str,
        parent: Optional["ClassNode"],
        index: Optional[int],
        meta: Optional[Mapping] = None,
        aliases: Optional[Iterable[str]] = None,
    ) -> None:
        if not name and parent is not None:
            raise ValueError("Only the root node is allowed an empty name.")

        super().__init__(name, parent, index, meta, aliases)
        self.classes: List["ClassNode"] = []
        self.tags: List[TagNode] = []
        self.virtuals: List[VirtualNode] = []

    @staticmethod
    def from_dict(
        name: str,
        data: Optional[Mapping],
        parent: Optional["ClassNode"] = None,
    ) -> "ClassNode":
        """Create a ClassNode from a dictionary representation."""
        if data is None:
            data = {}

        # Create node
        node = ClassNode(
            name,
            parent,
            data.get("index"),
            data.get("meta"),
            data.get("alias"),
        )

        # Create tags
        for tag_name, tag_data in (data.get("tags") or {}).items():
            node.add_tag(TagNode.from_dict(tag_name, tag_data, node))

        # Create children (which may reference tags)
        for class_name, class_data in (data.get("classes") or {}).items():
            node.add_class(ClassNode.from_dict(class_name, class_data, node))

        # Finally, create virtual nodes (which may reference tags and children)
        for virtual_name, virtual_description in (data.get("virtuals") or {}).items():
            try:
                description = node.parse_description(virtual_description)
                node.add_virtual(VirtualNode(virtual_name, node, description))
            except Exception as exc:
                raise ValueError(
                    f"Error parsing description {virtual_description!r} of virtual node '{node}/{virtual_name}'"
                ) from exc

        return node

    def to_dict(self):
        """Convert the ClassNode to a dictionary representation."""
        d = super().to_dict()
        if self.aliases:
            if isinstance(self.aliases, str) or len(self.aliases) > 1:
                d["alias"] = [a.pattern for a in self.aliases]
            else:
                d["alias"] = self.aliases[0].pattern  # type: ignore

        if self.classes:
            d["classes"] = {c.name: c.to_dict() for c in self.classes}

        if self.tags:
            d["tags"] = {t.name: t.to_dict() for t in self.tags}

        if self.virtuals:
            d["virtuals"] = {v.name: str(v.description) for v in self.virtuals}
        return d

    @property
    def real_children(self):
        return self.classes + self.tags

    def add_class(self, subclass: "ClassNode"):
        """Add a subclass."""
        # TODO: Make sure that the new node does not shadow an existing one
        self.classes.append(subclass)
        return subclass

    def add_tag(self, tag: TagNode):
        """Add a tag."""
        # TODO: Make sure that the new node does not shadow an existing one
        self.tags.append(tag)
        return tag

    def add_virtual(self, node: VirtualNode):
        """Add a VirtualNode."""
        # TODO: Make sure that the new node does not shadow an existing one
        self.virtuals.append(node)
        return node

    def _find_all_class(
        self, path: Sequence[str], with_alias=False, _base_specificy=0
    ) -> Iterable[Tuple["ClassNode", int]]:
        """Find all ClassNode instances matching the given path."""

        name, *tail = path

        specificy = self._matches_name(name, with_alias)

        # If self matches, descend with rest of the path
        if specificy:
            if tail:
                for subclass in self.classes:
                    yield from subclass._find_all_class(
                        tail, with_alias, specificy + _base_specificy
                    )
            else:
                yield (self, specificy + _base_specificy)

        # Also descend with the full path to find nodes that didn't specify the full path
        for subclass in self.classes:
            yield from subclass._find_all_class(path, with_alias, _base_specificy)

    def _find_all_tag(
        self, path: Sequence[str], with_alias=False
    ) -> Iterable[Tuple["TagNode", int]]:
        """Find all TagNode instances matching the given path."""

        for tag in self.tags:
            yield from tag._find_all_tag(path, with_alias)

        if self.parent is not None:
            yield from self.parent._find_all_tag(path, with_alias)

    def _find_all_virtual(
        self, path: Sequence[str], with_alias=False, _base_specificy=0
    ) -> Iterable[Tuple["VirtualNode", int]]:
        """Find all TagNode instances matching the given path."""

        name, *tail = path

        specificy = self._matches_name(name, with_alias)
        if specificy and tail:
            for subclass in self.classes:
                yield from subclass._find_all_virtual(
                    tail, with_alias, specificy + _base_specificy
                )

        if not tail:
            for virtual in self.virtuals:
                specificy = virtual._matches_name(name, with_alias)
                if specificy:
                    yield (virtual, specificy)

        if self.parent is not None:
            yield from self.parent._find_all_virtual(path, with_alias, _base_specificy)

    def find_class(
        self, name_or_path: Union[str, Sequence[str]], with_alias=False
    ) -> "ClassNode":
        """Find a class node by name or path."""
        name_or_path = _validate_name_or_path(name_or_path)

        return _best_match(
            self, name_or_path, self._find_all_class(name_or_path, with_alias)
        )

    def find_tag(self, name_or_path: Union[str, Sequence[str]]) -> TagNode:
        """Find specified tag in this node or its parents."""
        for tag in self.tags:
            try:
                return tag.find_tag(name_or_path)
            except NodeNotFoundError:
                pass

        if self.parent is not None:
            return self.parent.find_tag(name_or_path)

        raise NodeNotFoundError(
            f"Could not find {name_or_path} below {quote(self.format())}"
        )

    def get_applicable_virtuals(self):
        node = self
        while node is not None:
            yield from node.virtuals
            node = node.parent

    def find_virtual(
        self, name_or_path: Union[str, Sequence[str]], with_alias=False
    ) -> VirtualNode:
        """Find specified virtual node in this node or its parents."""
        return _best_match(
            self, name_or_path, self._find_all_virtual(name_or_path, with_alias)
        )

    @fill_in_doc(_doc_fields)
    def find_any_node(
        self, name_or_path: Union[str, Sequence[str]], with_alias=False
    ) -> BaseNode:
        """
        Find any node (class, tag or virtual) in relation to self.

        Args:
            names (iterable of str): ...
            {with_alias_arg}

        Returns:
            Description: The parsed Description.
        """

        name_or_path = _validate_name_or_path(name_or_path)

        if not with_alias:
            try:
                return self.find_class(name_or_path)
            except NodeNotFoundError:
                pass

            try:
                return self.find_tag(name_or_path)
            except NodeNotFoundError:
                pass

            try:
                return self.find_virtual(name_or_path)
            except NodeNotFoundError:
                pass

            raise NodeNotFoundError(
                f"Could not find {name_or_path} below {quote(self.format())}"
            )

        # Find all matching nodes
        matches = []
        matches.extend(self._find_all_class(name_or_path, with_alias=with_alias))
        matches.extend(self._find_all_tag(name_or_path, with_alias=with_alias))
        matches.extend(self._find_all_virtual(name_or_path, with_alias=with_alias))

        # Sort matches by specificy
        return _best_match(self, name_or_path, matches)

    def format_tree(self, extra=None, virtuals=False) -> str:
        """Format the ClassNode and its children as a tree."""
        name = self.name

        attrs = []
        if self.index is not None:
            attrs.append(f"index={self.index}")

        if self.meta:
            attrs.append(f"meta={self.meta}")

        if attrs:
            name = name + " (" + (", ".join(attrs)) + ")"

        if self.index is not None and extra is not None:
            name += f" [{extra[self.index]}]"

        lines = [f"/{name}"]

        for tag in self.tags:
            lines.append(indent(tag.format_tree(extra), "  "))

        if virtuals:
            for virtual in self.virtuals:
                lines.append(indent(virtual.format_tree(), "  "))

        for subclass in self.classes:
            lines.append(indent(subclass.format_tree(extra, virtuals), "  "))

        return "\n".join(lines)

    def find_real_node(
        self, name_or_path: Union[str, Sequence[str]], with_alias=False
    ) -> Union["ClassNode", "TagNode"]:
        """Find a real node (ClassNode or TagNode) by name or path."""
        if not with_alias:
            try:
                return self.find_class(name_or_path)
            except NodeNotFoundError:
                pass

            try:
                return self.find_tag(name_or_path)
            except NodeNotFoundError:
                pass

            raise NodeNotFoundError(f"{name_or_path} (anchor={quote(self.format())})")
        else:
            matches = []
            matches.extend(self._find_all_class(name_or_path, with_alias=with_alias))
            matches.extend(self._find_all_tag(name_or_path, with_alias=with_alias))
            return _best_match(self, name_or_path, matches)

    def parse_description(
        self,
        description: str,
        with_alias=False,
        on_conflict: Literal["replace", "raise"] = "replace",
    ):
        """
        Parse a description string into a Description.

        Args:
            description (str): The description string to parse.
            with_alias (bool, optional): Whether to consider aliases. Defaults to False.
            {on_conflict_arg}

        Returns:
            Description: The parsed Description.
        """

        return Description.from_string(
            self,
            description,
            with_alias=with_alias,
            on_conflict=on_conflict,
        )

    def union(self, other: "ClassNode"):
        """Return the union of the current node with another node."""
        if self in other.precursors:
            return other

        if other in self.precursors:
            return self

        raise ValueError(f"{self} and {other} are incompatible")

    @functools.cached_property
    def rivalling_children(self):
        """The set of directly rivalling descendants."""

        result: List[ClassNode] = []
        for subclass in self.classes:
            if subclass.index is not None:
                result.append(subclass)
            else:
                result.extend(subclass.rivalling_children)

        return result

    def negate(self, negate: bool = True) -> Descriptor:
        if negate and self.parent is None:
            return NeverDescriptor()

        return super().negate(negate)

    def __le__(self, other) -> bool:
        if isinstance(other, (TagNode, ClassNode)):
            return self in other.precursors

        if isinstance(other, NegatedRealNode):
            return False

        return super().__le__(other)


TOnConflictLiteral = Literal["replace", "raise", "skip"]


class ConflictError(Exception):
    """Exception raised when descriptors are incompatible."""

    pass


@fill_in_doc(_doc_fields)
class Description:
    """
    Represents a PolyTaxo description.

    Descriptions are used to describe an object using a sequence of descriptors.

    Args:
        {anchor_arg}
        qualifiers (Iterable[Descriptor], optional): Additional parts of the description.
    """

    def __init__(
        self,
        anchor: "ClassNode",
        qualifiers: Optional[Iterable[Descriptor]] = None,
    ) -> None:
        self.anchor: "ClassNode" = anchor
        self.qualifiers: List[Descriptor] = []

        if qualifiers is not None:
            self.update(qualifiers, on_conflict="raise")

    def _parse_description_tokens(
        self,
        tokens: Iterator,
        with_alias: bool,
        on_conflict: TOnConflictLiteral,
    ) -> List[Descriptor]:
        negate = False
        descriptors: List[Descriptor] = []

        while True:
            try:
                token = next(tokens)
            except StopIteration:
                return descriptors
            if token == "!":
                negate = not negate
                continue
            elif isinstance(token, tuple):
                # We have a tuple of names, let's find the corresponding node
                node = self.anchor.find_real_node(token, with_alias=with_alias)
                d = node.negate(negate)
                descriptors.append(d)
                self.add(d, on_conflict=on_conflict)
                negate = False
            else:
                raise ValueError(f"Unexpected token in description: {token!r}")

    @staticmethod
    @fill_in_doc(_doc_fields)
    def from_string(
        anchor: ClassNode,
        description_str: str,
        with_alias=False,
        on_conflict: TOnConflictLiteral = "replace",
    ) -> "Description":
        """
        Parse a textual description into a Description object.

        Args:
            {anchor_arg}
        """
        tokens = iter(tokenize(description_str))
        d = Description(anchor)
        d._parse_description_tokens(
            tokens,
            with_alias=with_alias,
            on_conflict=on_conflict,
        )
        return d

    # @overload
    # @staticmethod
    # def from_lineage(
    #     anchor: ClassNode,
    #     names: Iterable[str],
    #     *,
    #     with_alias: bool,
    #     on_conflict: TOnConflictLiteral,
    #     ignore_unmatched_intermediaries: bool,
    #     return_unmatched_suffix: Literal[True],
    # ) -> Tuple["Description", List[str]]: ...

    # @overload
    # @staticmethod
    # def from_lineage(
    #     anchor: ClassNode,
    #     names: Iterable[str],
    #     *,
    #     with_alias: bool,
    #     on_conflict: TOnConflictLiteral,
    #     ignore_unmatched_intermediaries: bool,
    #     return_unmatched_suffix: Literal[False],
    # ) -> "Description": ...

    @staticmethod
    @fill_in_doc(_doc_fields)
    def from_lineage(
        anchor: ClassNode,
        names: Iterable[str],
        *,
        with_alias: bool = False,
        on_conflict: TOnConflictLiteral = "replace",
        ignore_unmatched_intermediaries: bool = False,
        return_unmatched_suffix: bool = False,
    ):
        """
        Parse a sequence of names into a Description object.

        This can be used to transform a strictly hierarchical representation into a PolyTaxo Description.

        Args:
            {anchor_arg}
            names (iterable of str): ...
            {with_alias_arg}
            {on_conflict_arg}
            {ignore_unmatched_intermediaries_arg}

        Returns:
            Description: The parsed Description.
        """

        description = Description(anchor)

        unmatched_names: List[str] = []

        for name in names:
            try:
                # TODO: This should rather be parse_name() -> Descriptor | Description
                node = description.anchor.find_any_node(name, with_alias)
            except NodeNotFoundError:
                if ignore_unmatched_intermediaries:
                    unmatched_names.append(name)
                    continue
                raise

            unmatched_names.clear()

            if isinstance(node, Descriptor):
                description.add(node, on_conflict=on_conflict)
            elif isinstance(node, VirtualNode):
                description.add(node.description, on_conflict=on_conflict)
            else:
                raise ValueError(f"Unexpected node: {node}")

        if return_unmatched_suffix:
            return description, unmatched_names

        if unmatched_names:
            raise ValueError(f"Unmatched suffix: {'/'.join(unmatched_names)}")

        return description

    @property
    def descriptors(self) -> Sequence[Descriptor]:
        """Return all descriptors (anchor + qualifiers)."""
        return [self.anchor] + self.qualifiers

    def copy(self) -> "Description":
        """Create a copy of the current Description."""
        return Description(self.anchor, self.qualifiers)

    def _add_poly_description(
        self,
        other: "Description",
        on_conflict: TOnConflictLiteral,
    ):
        self.add(other.anchor, on_conflict)

        for qualifier in other.qualifiers:
            self.add(qualifier, on_conflict)

    def _add_class(
        self,
        other: "ClassNode",
        on_conflict: TOnConflictLiteral,
    ):
        if self.anchor <= other:
            # anchor more general than other: replace
            self.anchor = other
            return

        if other <= self.anchor:
            # other is more general than self: do nothing
            return

        # Otherwise, other and self.anchor are in conflict
        if on_conflict == "replace":
            self.anchor = other
            return
        if on_conflict == "skip":
            return

        raise ConflictError(f"{self.anchor} and {other} are incompatible")

    def _add_tag(
        self,
        other: "TagNode",
        on_conflict: TOnConflictLiteral,
    ):
        # Check if other is already implied
        if other <= self:
            return

        # Add parent class of tag
        self._add_class(other.parent_class, on_conflict=on_conflict)

        # Remove existing qualifiers that are implied by `other`
        qualifiers = [q for q in self.qualifiers if not (q <= other)]

        if on_conflict == "replace":
            # Remove existing qualifiers that contradict `other` (!q is implied by other)
            qualifiers = [q for q in qualifiers if not (q.negate() <= other)]
        else:
            for q in qualifiers:
                if q.negate() <= other:
                    if on_conflict == "skip":
                        return

                    raise ConflictError(f"{q} and {other} are incompatible")

        self.qualifiers = qualifiers + [other]

    def _add_negated_class(
        self,
        other: "NegatedRealNode[ClassNode]",
        on_conflict: TOnConflictLiteral,
    ):
        # Check if other is already implied
        if other <= self:
            return

        if other.node <= self.anchor:
            if on_conflict == "raise":
                raise ConflictError(f"{self.anchor} and {other} are incompatible")

            if on_conflict == "skip":
                return

            # Delete the precluded portion of the anchor
            assert other.node.parent is not None
            self.anchor = other.node.parent

        # Remove existing qualifiers implied by other
        qualifiers = [q for q in self.qualifiers if not (q <= other)]

        self.qualifiers = qualifiers + [other]

    def _add_negated_tag(
        self,
        other: "NegatedRealNode[TagNode]",
        on_conflict: TOnConflictLiteral,
    ):
        # Check if other is already implied
        if other <= self:
            return

        qualifiers = self.qualifiers

        if on_conflict == "replace":
            # Remove existing qualifiers that contradict `other`
            qualifiers = [q for q in qualifiers if not (q.negate() <= other)]
        else:
            for q in qualifiers:
                if q.negate() <= other:
                    if on_conflict == "skip":
                        return

                    raise ConflictError(f"{q} and {other} are incompatible")

        # Remove existing qualifiers that are implied by `other`
        qualifiers = [q for q in qualifiers if not (q <= other)]

        self.qualifiers = qualifiers + [other]

    def add(
        self,
        other: Union["Description", Descriptor],
        on_conflict: TOnConflictLiteral = "replace",
    ) -> "Description":
        """Add a descriptor or description to the current description."""
        if on_conflict not in ("replace", "raise", "skip"):
            raise ValueError(f"Unexpected value for on_conflict: {on_conflict}")

        if isinstance(other, Description):
            self._add_poly_description(other, on_conflict)

        elif isinstance(other, ClassNode):
            self._add_class(other, on_conflict)

        elif isinstance(other, TagNode):
            self._add_tag(other, on_conflict)

        elif isinstance(other, NegatedRealNode):
            if isinstance(other.node, ClassNode):
                self._add_negated_class(other, on_conflict)

            elif isinstance(other.node, TagNode):
                self._add_negated_tag(other, on_conflict)

            else:
                raise ValueError(
                    f"Unexpected type of NegatedQualifier.node: {type(other.node)}"
                )
        else:
            raise ValueError(f"Unexpected type of other: {type(other)}")

        return self

    def update(
        self,
        descriptors: Iterable[Descriptor],
        on_conflict: TOnConflictLiteral = "replace",
    ):
        for descriptor in descriptors:
            self.add(descriptor, on_conflict=on_conflict)

        return self

    def _remove_description(self, other: "Description"):
        for descriptor in other.descriptors:
            self.remove(descriptor)

    def _remove_descriptor(self, other: Descriptor):
        # Replace qualifiers that imply "other" with their nearest ancestor that does not imply "other"

        if other <= self.anchor:
            # Delete the precluded portion of the anchor
            if other.parent is None:
                raise ValueError("Cannot remove root")

            self.anchor = other.parent

        new_qualifiers = []

        for q in self.qualifiers:
            if other <= q:
                # "other" is implied
                if (
                    isinstance(other, TagNode)
                    and isinstance(other.parent, TagNode)
                    and isinstance(other.parent.parent, TagNode)
                ):
                    # if other has a parent of the same type (TagNode): keep "other.parent" as the nearest ancestor that does not imply "other"
                    # (But don't keep group tags that don't carry information themselves.)
                    new_qualifiers.append(other.parent)
            else:
                # "other" not implied: keep "q"
                new_qualifiers.append(q)

        self.qualifiers = new_qualifiers

    def remove(self, other: Union["Description", Descriptor]):
        """Remove a descriptor or description from the current description."""
        if isinstance(other, Description):
            self._remove_description(other)

        elif isinstance(other, Descriptor):
            self._remove_descriptor(other)

        else:
            raise ValueError(f"Unexpected type of other: {type(other)}")

        return self

    def format(self, anchor: ClassNode | None = None):
        # Sort qualifiers alphabetically for stable lookup
        qualifiers = sorted(
            [q.format(anchor=self.anchor, quoted=True) for q in self.qualifiers]
        )

        return " ".join([self.anchor.format(anchor, quoted=True)] + qualifiers)

    def __le__(self, other) -> bool:
        if not isinstance(other, Description):
            return NotImplemented

        for d in self.descriptors:
            if not (d <= other):
                return False

        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Description):
            return False

        return set(self.descriptors) == set(other.descriptors)

    def __hash__(self) -> int:
        return hash(frozenset(self.descriptors))

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self!s}>"

    def __str__(self) -> str:
        return self.format()
