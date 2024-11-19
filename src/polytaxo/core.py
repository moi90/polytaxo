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
    "anchor_arg": """anchor (PrimaryNode): The starting point for interpreting descriptors.""",
    "on_conflict_arg": """on_conflict ("replace", "raise", or "skip", optional): Conflict resolution strategy. Defaults to "replace".""",
    "ignore_unmatched_intermediaries_arg": """ignore_unmatched_intermediaries (bool, optional): Whether to ignore unmatched intermediaries. Defaults to False.""",
    "with_alias_arg": """with_alias (bool, optional): Whether to consider aliases. Defaults to False.""",
}


class NodeNotFoundError(Exception):
    """Exception raised when a node is not found."""

    pass


class TreeDescriptor(Descriptor):
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
    """Base class for all nodes."""

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
        """Return a tuple of sibling nodes."""
        if self.parent is None:
            return tuple()

        if isinstance(self.parent, (PrimaryNode, TagNode)):
            return tuple(c for c in self.parent.children if c is not self)

        return tuple()

    @functools.cached_property
    def path(self):
        """Return the path from the root to this node as a tuple of names."""
        return tuple(n.name for n in self.precursors)

    def format(self, anchor: Union["PrimaryNode", None] = None, quoted=False) -> str:
        """Format the node as a string relative to an optional anchor."""
        precursors = self.precursors

        if anchor is not None:
            # TagNodes are displayed relative to the next precursor of anchor
            if isinstance(self, TagNode):
                # Find first PrimaryNode in precursors
                tag_anchor = self.primary_parent
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
        if name == self.name:
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


class RealNode(BaseNode, TreeDescriptor):
    """Base class for real nodes (PrimaryNode, TagNode)."""

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

    def _get_all_children(self):
        raise NotImplementedError()

    def _include_in_binary(self):
        """Determine if the node should be included in the binary output."""
        return (type(self.parent) == type(self)) or not self._get_all_children()

    def fill_indices(self, index_provider: IndexProvider):
        """Fill the indices for the node and its children."""
        if self.index is None and self._include_in_binary():
            self.index = index_provider.take()
        for child in self._get_all_children():
            if isinstance(child, RealNode):
                child.fill_indices(index_provider)

    def walk(self) -> Iterator["RealNode"]:
        """Walk through the node and its children."""
        yield self
        for child in self._get_all_children():
            if isinstance(child, RealNode):
                yield from child.walk()

    def negate(self, negate: bool = True) -> Descriptor:
        if negate:
            return NegatedRealNode(self)
        return self


TRealNode = TypeVar("TRealNode", bound=RealNode)


class NegatedRealNode(TreeDescriptor, Generic[TRealNode]):
    """Represents a negated real node."""

    def __init__(self, node: TRealNode) -> None:
        # We cannot negate the root
        if node.parent is None:
            raise ValueError(f"Root node cannot be negated")

        self.node = node

    def format(self, anchor: Union["PrimaryNode", None] = None, quoted=False) -> str:
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

        if isinstance(self.node, PrimaryNode):
            if isinstance(other, TagNode):
                return False

            if isinstance(other, PrimaryNode):
                # !A <= B if A is a sibling of any precursor of B
                for n in other.precursors[::-1]:
                    if any(s <= self.node for s in n.siblings):
                        return True

                return False

        if isinstance(self.node, TagNode):
            if isinstance(other, PrimaryNode):
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

    parent: Union["PrimaryNode", "TagNode"]

    def __init__(
        self,
        name: str,
        parent: Union["PrimaryNode", "TagNode"],
        index: Optional[int],
        meta: Optional[Mapping] = None,
        aliases: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(name, parent, index, meta, aliases)

        self.children: List["TagNode"] = []

    @staticmethod
    def from_dict(
        name, data: Optional[Mapping], parent: Union["PrimaryNode", "TagNode"]
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

        for child_name, child_data in data.get("children", {}).items():
            tag_node.add_child(TagNode.from_dict(child_name, child_data, tag_node))

        return tag_node

    def to_dict(self) -> Dict[str, Any]:
        """Convert the TagNode to a dictionary representation."""
        d = super().to_dict()
        if self.children:
            d["children"] = {c.name: c.to_dict() for c in self.children}

        return d

    def add_child(self, node: "TagNode"):
        """Add a child TagNode."""
        self.children.append(node)

    def _get_all_children(self):
        return self.children

    @functools.cached_property
    def primary_parent(self) -> "PrimaryNode":
        """Return the primary parent (first parent that is a PrimaryNode) of the TagNode."""
        parent = self.parent
        while parent is not None:
            if isinstance(parent, PrimaryNode):
                return parent
            parent = parent.parent

        raise ValueError(
            f"Tag without a primary parent: {self.name}"
        )  # pragma: no cover

    def _find_all_tag(
        self, path: Sequence[str], with_alias=False, _base_specificy=0
    ) -> Iterable[Tuple["TagNode", int]]:
        """Find all TagNode instances matching the given path."""

        name, *tail = path

        specificy = self._matches_name(name, with_alias)

        # If self matches, descend with rest of the path
        if specificy:
            if tail:
                for child in self.children:
                    yield from child._find_all_tag(
                        tail, with_alias, specificy + _base_specificy
                    )
            else:
                yield (self, specificy + _base_specificy)

        # Also descend with the full path to find nodes that didn't specify the full path
        for child in self.children:
            yield from child._find_all_tag(path, with_alias, _base_specificy)

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

        lines = [f"{name}:"]

        for child in self.children:
            lines.append(indent(child.format_tree(extra_info), "  "))

        return "\n".join(lines)

    @functools.cached_property
    def rivalling_children(self):
        """The set of directly rivalling descendants."""

        result: List[TagNode] = []
        for child in self.children:
            if child.index is not None:
                result.append(child)
            else:
                result.extend(child.rivalling_children)

        return result

    def __le__(self, other) -> bool:
        if isinstance(other, TagNode):
            return self in other.precursors

        if isinstance(other, PrimaryNode):
            return False

        if isinstance(other, NegatedRealNode):
            return False

        return super().__le__(other)


class VirtualNode(BaseNode):
    """Represents a virtual node."""

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
        return f"{self.name} -> {self.description!s}"

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.format_tree()}>"


class PrimaryNode(RealNode):
    """Represents a primary node."""

    parent: Optional["PrimaryNode"]

    def __init__(
        self,
        name: str,
        parent: Optional["PrimaryNode"],
        index: Optional[int],
        meta: Optional[Mapping] = None,
        aliases: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(name, parent, index, meta, aliases)
        self.children: List["PrimaryNode"] = []
        self.tags: List[TagNode] = []
        self.virtuals: List[VirtualNode] = []

    @staticmethod
    def from_dict(
        name,
        data: Optional[Mapping],
        parent: Optional["PrimaryNode"] = None,
    ) -> "PrimaryNode":
        """Create a PrimaryNode from a dictionary representation."""
        if data is None:
            data = {}

        # Create node
        node = PrimaryNode(
            name,
            parent,
            data.get("index"),
            data.get("meta"),
            data.get("alias"),
        )

        # Create tags
        for tag_name, tag_data in data.get("tags", {}).items():
            node.add_tag(TagNode.from_dict(tag_name, tag_data, node))

        # Create children (which may reference tags)
        for child_name, child_data in data.get("children", {}).items():
            node.add_child(PrimaryNode.from_dict(child_name, child_data, node))

        # Finally, create virtual nodes (which may reference tags and children)
        for virtual_name, virtual_description in data.get("virtuals", {}).items():
            try:
                description = node.parse_description(virtual_description)
                node.add_virtual(VirtualNode(virtual_name, node, description))
            except Exception as exc:
                raise ValueError(
                    f"Error parsing description {virtual_description!r} of virtual node '{node}/{virtual_name}'"
                ) from exc

        return node

    def to_dict(self):
        """Convert the PrimaryNode to a dictionary representation."""
        d = super().to_dict()
        if self.aliases:
            if isinstance(self.aliases, str) or len(self.aliases) > 1:
                d["alias"] = [a.pattern for a in self.aliases]
            else:
                d["alias"] = self.aliases[0].pattern  # type: ignore

        if self.children:
            d["children"] = {c.name: c.to_dict() for c in self.children}

        if self.tags:
            d["tags"] = {t.name: t.to_dict() for t in self.tags}

        if self.virtuals:
            d["virtuals"] = {v.name: str(v.description) for v in self.virtuals}
        return d

    def _get_all_children(self):
        return self.children + self.tags + self.virtuals

    def add_child(self, node: "PrimaryNode"):
        """Add a child PrimaryNode."""
        # TODO: Make sure that the new node does not shadow an existing one
        self.children.append(node)
        return node

    def add_tag(self, node: TagNode):
        """Add a TagNode."""
        # TODO: Make sure that the new node does not shadow an existing one
        self.tags.append(node)
        return node

    def add_virtual(self, node: VirtualNode):
        """Add a VirtualNode."""
        # TODO: Make sure that the new node does not shadow an existing one
        self.virtuals.append(node)
        return node

    def _find_all_primary(
        self, path: Sequence[str], with_alias=False, _base_specificy=0
    ) -> Iterable[Tuple["PrimaryNode", int]]:
        """Find all PrimaryNode instances matching the given path."""

        name, *tail = path

        specificy = self._matches_name(name, with_alias)

        # If self matches, descend with rest of the path
        if specificy:
            if tail:
                for child in self.children:
                    yield from child._find_all_primary(
                        tail, with_alias, specificy + _base_specificy
                    )
            else:
                yield (self, specificy + _base_specificy)

        # Also descend with the full path to find nodes that didn't specify the full path
        for child in self.children:
            yield from child._find_all_primary(path, with_alias, _base_specificy)

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
            for child in self.children:
                yield from child._find_all_virtual(
                    tail, with_alias, specificy + _base_specificy
                )

        if not tail:
            for virtual in self.virtuals:
                specificy = virtual._matches_name(name, with_alias)
                if specificy:
                    yield (virtual, specificy)

        if self.parent is not None:
            yield from self.parent._find_all_virtual(path, with_alias, _base_specificy)

    def _find_primary(self, name, with_alias=False) -> "PrimaryNode":
        """Return the first PrimaryNode from the current subtree with the given name."""

        matches = self._find_all_primary(name, with_alias)

        if not matches:
            raise NodeNotFoundError(
                f"Could not find {name} below {quote(self.format())}"
            )

        # If we're matching with alias, multiple children can produce a match.
        # We therefore select the shortest match
        matches.sort(key=lambda n: len(n.precursors))

        return matches[0]

    def find_primary(
        self, name_or_path: Union[str, Sequence[str]], with_alias=False
    ) -> "PrimaryNode":
        """Find a primary node by name or path."""
        name_or_path = _validate_name_or_path(name_or_path)

        return _best_match(
            self, name_or_path, self._find_all_primary(name_or_path, with_alias)
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
        Find any node (primary, tag or virtual) in relation to self.

        Args:
            names (iterable of str): ...
            {with_alias_arg}

        Returns:
            PolyDescription: The parsed PolyDescription.
        """

        name_or_path = _validate_name_or_path(name_or_path)

        if not with_alias:
            try:
                return self.find_primary(name_or_path)
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
        matches.extend(self._find_all_primary(name_or_path, with_alias=with_alias))
        matches.extend(self._find_all_tag(name_or_path, with_alias=with_alias))
        matches.extend(self._find_all_virtual(name_or_path, with_alias=with_alias))

        print(f"matches for {name_or_path}", matches)

        # Sort matches by specificy
        return _best_match(self, name_or_path, matches)

    def format_tree(self, extra=None, virtuals=False) -> str:
        """Format the PrimaryNode and its children as a tree."""
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

        lines = [f"{name}::"]

        for child in self.children:
            lines.append(indent(child.format_tree(extra, virtuals), "  "))

        for tag in self.tags:
            lines.append(indent(tag.format_tree(extra), "  "))

        if virtuals:
            for virtual in self.virtuals:
                lines.append(indent(virtual.format_tree(), "  "))

        return "\n".join(lines)

    def find_real_node(
        self, name_or_path: Union[str, Sequence[str]], with_alias=False
    ) -> Union["PrimaryNode", "TagNode"]:
        """Find a real node (PrimaryNode or TagNode) by name or path."""
        if not with_alias:
            try:
                return self.find_primary(name_or_path)
            except NodeNotFoundError:
                pass

            try:
                return self.find_tag(name_or_path)
            except NodeNotFoundError:
                pass

            raise NodeNotFoundError(f"{name_or_path} (anchor={quote(self.format())})")
        else:
            matches = []
            matches.extend(self._find_all_primary(name_or_path, with_alias=with_alias))
            matches.extend(self._find_all_tag(name_or_path, with_alias=with_alias))
            return _best_match(self, name_or_path, matches)

    def parse_description(
        self,
        description: str,
        with_alias=False,
        on_conflict: Literal["replace", "raise"] = "replace",
    ):
        """
        Parse a description string into a PolyDescription.

        Args:
            description (str): The description string to parse.
            with_alias (bool, optional): Whether to consider aliases. Defaults to False.
            {on_conflict_arg}

        Returns:
            PolyDescription: The parsed PolyDescription.
        """

        return Description.from_string(
            self,
            description,
            with_alias=with_alias,
            on_conflict=on_conflict,
        )

    def union(self, other: "PrimaryNode"):
        """Return the union of the current node with another node."""
        if self in other.precursors:
            return other

        if other in self.precursors:
            return self

        raise ValueError(f"{self} and {other} are incompatible")

    @functools.cached_property
    def rivalling_children(self):
        """The set of directly rivalling descendants."""

        result: List[PrimaryNode] = []
        for child in self.children:
            if child.index is not None:
                result.append(child)
            else:
                result.extend(child.rivalling_children)

        return result

    def negate(self, negate: bool = True) -> Descriptor:
        if negate and self.parent is None:
            return NeverDescriptor()

        return super().negate(negate)

    def __le__(self, other) -> bool:
        if isinstance(other, (TagNode, PrimaryNode)):
            return self in other.precursors

        if isinstance(other, NegatedRealNode):
            return False

        return super().__le__(other)


TOnConflictLiteral = Literal["replace", "raise", "skip"]


class ConflictError(Exception):
    """Exception raised when descriptors are incompatible."""

    pass


class Description:
    """Represents a polyhierarchical description."""

    def __init__(
        self,
        anchor: "PrimaryNode",
        qualifiers: Optional[Iterable[Descriptor]] = None,
    ) -> None:
        self.anchor: "PrimaryNode" = anchor
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

    @fill_in_doc(_doc_fields)
    @staticmethod
    def from_string(
        anchor: PrimaryNode,
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

    @fill_in_doc(_doc_fields)
    @staticmethod
    def from_lineage(
        anchor: PrimaryNode,
        names: Iterable[str],
        with_alias=False,
        on_conflict: TOnConflictLiteral = "replace",
        ignore_unmatched_intermediaries=False,
    ) -> "Description":
        """
        Parse a sequence of names into a Description object.

        Args:
            {anchor_arg}
            names (iterable of str): ...
            {with_alias_arg}
            {on_conflict_arg}
            {ignore_unmatched_intermediaries_arg}

        Returns:
            PolyDescription: The parsed PolyDescription.
        """

        description = Description(anchor)

        unmatched_names = []

        for name in names:
            try:
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

        if unmatched_names:
            raise ValueError(f"Unmatched suffix: {'/'.join(unmatched_names)}")

        return description

    @property
    def descriptors(self) -> Sequence[Descriptor]:
        """Return all descriptors (anchor + qualifiers)."""
        return [self.anchor] + self.qualifiers

    def to_binary_raw(self) -> Mapping["RealNode", bool]:
        """Convert the description to a binary representation of nodes and their active state."""
        map: Mapping[RealNode, bool] = {}

        def handle_positive(node: RealNode):
            # Activate node and all precursors
            for precursor in node.precursors:
                if not isinstance(precursor, node.__class__):
                    continue

                if precursor._include_in_binary():
                    map[precursor] = True

                # Deactivate all siblings
                for sibling in precursor.siblings:
                    if not isinstance(sibling, node.__class__):
                        continue

                    if sibling._include_in_binary():
                        map[sibling] = False

        def handle_negative(node: RealNode):
            # Deactivate node and all successors
            for successor in node.walk():
                if not isinstance(successor, node.__class__):
                    continue

                if successor._include_in_binary():
                    map[successor] = False

        handle_positive(self.anchor)

        for qualifier in self.qualifiers:
            if isinstance(qualifier, RealNode):
                handle_positive(qualifier)
            elif isinstance(qualifier, NegatedRealNode):
                handle_negative(qualifier.node)
            else:
                raise ValueError(f"Unknown qualifier type: {type(qualifier)}")

        return map

    def to_binary_str(self) -> Mapping[str, bool]:
        """Convert the binary representation to a string representation."""
        return {str(k): v for k, v in self.to_binary_raw().items()}

    def to_multilabel(self, n_labels=None, fill_na: Any = -1) -> List:
        """Transform the description to a list of target values."""
        int_map = {
            n.index: v for n, v in self.to_binary_raw().items() if n.index is not None
        }

        if n_labels is None:
            n_labels = max(int_map.keys()) + 1

        return [int_map.get(i, fill_na) for i in range(n_labels)]

    def copy(self) -> "Description":
        """Create a copy of the current PolyDescription."""
        return Description(self.anchor, self.qualifiers)

    def _add_poly_description(
        self,
        other: "Description",
        on_conflict: TOnConflictLiteral,
    ):
        self.add(other.anchor, on_conflict)

        for qualifier in other.qualifiers:
            self.add(qualifier, on_conflict)

    def _add_primary(
        self,
        other: "PrimaryNode",
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

        # Add primary parent of tag
        self._add_primary(other.primary_parent, on_conflict=on_conflict)

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

    def _add_negated_primary(
        self,
        other: "NegatedRealNode[PrimaryNode]",
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
        """Add a descriptor or poly description to the current description."""
        if on_conflict not in ("replace", "raise", "skip"):
            raise ValueError(f"Unexpected value for on_conflict: {on_conflict}")

        if isinstance(other, Description):
            self._add_poly_description(other, on_conflict)

        elif isinstance(other, PrimaryNode):
            self._add_primary(other, on_conflict)

        elif isinstance(other, TagNode):
            self._add_tag(other, on_conflict)

        elif isinstance(other, NegatedRealNode):
            if isinstance(other.node, PrimaryNode):
                self._add_negated_primary(other, on_conflict)

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
        """Remove a descriptor or poly description from the current description."""
        if isinstance(other, Description):
            self._remove_description(other)

        elif isinstance(other, Descriptor):
            self._remove_descriptor(other)

        else:
            raise ValueError(f"Unexpected type of other: {type(other)}")

        return self

    def format(self, anchor: PrimaryNode | None = None):
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
