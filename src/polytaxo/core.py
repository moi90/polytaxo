import fnmatch
import functools
import itertools
import shlex
from textwrap import indent
from typing import (
    Any,
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

from .descriptor import Descriptor, NeverDescriptor
from .parser import _tokenize_expression_str


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

        return NotImplemented


class BaseNode:
    """Base class for all nodes."""

    def __init__(
        self,
        name: str,
        parent: Optional["BaseNode"],
    ) -> None:
        self.name = name
        self.parent = parent

    def set_parent(self, parent: Optional["BaseNode"]):
        """Set the parent node."""
        self.parent = parent
        return self

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

    def format(self, anchor: Union["PrimaryNode", None] = None) -> str:
        """Format the node as a string relative to an optional anchor."""
        precursors = self.precursors

        if anchor is not None:
            # TagNodes are displayed relative to the next precursor of anchor
            if isinstance(self, TagNode):
                # Find first PrimaryNode in precursors
                tag_anchor = self.primary_parent
                if tag_anchor in anchor.precursors:
                    anchor = tag_anchor

            if anchor != self:
                try:
                    i = precursors.index(anchor)
                except:
                    pass
                else:
                    precursors = precursors[i + 1 :]

        def build():
            sep = ""
            for n in precursors:
                yield sep + n.name
                if isinstance(n, TagNode):
                    sep = ":"
                else:
                    sep = "/"

        return "".join(build())


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
    ) -> None:
        super().__init__(name, parent)
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

    def format(self, anchor: Union["PrimaryNode", None] = None) -> str:
        """Format the negated node as a string."""
        return f"!{self.node.format(anchor)}"

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
    ) -> None:
        super().__init__(name, parent, index, meta)

        self.children: List["TagNode"] = []

    @staticmethod
    def from_dict(
        name, data: Optional[Mapping], parent: Union["PrimaryNode", "TagNode"]
    ) -> "TagNode":
        """Create a TagNode from a dictionary representation."""
        if data is None:
            data = {}

        tag_node = TagNode(name, parent, data.get("index"), data.get("meta"))

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
    def primary_parent(self) -> Optional["PrimaryNode"]:
        """Return the primary parent (first parent that is a PrimaryNode) of the TagNode."""
        parent = self.parent
        while parent is not None:
            if isinstance(parent, PrimaryNode):
                return parent
            parent = parent.parent
        return None

    def _find_tag(self, name) -> "TagNode":
        if name == self.name:
            return self

        for child in self.children:
            try:
                return child._find_tag(name)
            except NodeNotFoundError:
                pass

        raise NodeNotFoundError(name)

    def find_tag(self, name_or_path: Union[str, Iterable[str]]) -> "TagNode":
        """Find a tag by name or path."""
        if isinstance(name_or_path, str):
            name_or_path = (name_or_path,)

        node = self
        while name_or_path:
            head, *name_or_path = name_or_path
            node = node._find_tag(head)

        return node

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
        alias: Optional[Iterable[str]] = None,
        meta: Optional[Mapping] = None,
    ) -> None:
        super().__init__(name, parent, index, meta)
        self.children: List["PrimaryNode"] = []
        self.tags: List[TagNode] = []
        self.virtuals: List[VirtualNode] = []

        if alias is None:
            alias = tuple()
        elif isinstance(alias, str):
            alias = (alias,)
        self.alias = tuple(alias)

    @staticmethod
    def from_dict(
        name, data: Optional[Mapping], parent: Optional["PrimaryNode"]
    ) -> "PrimaryNode":
        """Create a PrimaryNode from a dictionary representation."""
        if data is None:
            data = {}

        # Create node
        node = PrimaryNode(
            name, parent, data.get("index"), data.get("alias"), data.get("meta")
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
                if isinstance(virtual_description, str):
                    description = node.parse_description(virtual_description)
                else:
                    description = node.get_description(virtual_description)
                virtual_node = VirtualNode(virtual_name, node, description)
                node.add_virtual(virtual_node)
            except Exception as exc:
                raise ValueError(
                    f"Error parsing description {virtual_description!r} of virtual node '{node}/{virtual_name}'"
                ) from exc

        return node

    def to_dict(self):
        """Convert the PrimaryNode to a dictionary representation."""
        d = super().to_dict()
        if self.alias:
            if isinstance(self.alias, str) or len(self.alias) > 1:
                d["alias"] = self.alias
            else:
                d["alias"] = self.alias[0]  # type: ignore

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
        self.children.append(node)
        return node

    def add_tag(self, node: TagNode):
        """Add a TagNode."""
        self.tags.append(node)
        return node

    def add_virtual(self, node: VirtualNode):
        """Add a VirtualNode."""
        self.virtuals.append(node)
        return node

    def _matches_name(self, name: str, with_alias: bool):
        """Check if the node matches the given name or alias."""
        if name == self.name:
            return True

        if with_alias:
            for alias in self.alias:
                if fnmatch.fnmatch(name, alias):
                    return True

        return False

    def find_all_primary(self, name: str, with_alias=False) -> List["PrimaryNode"]:
        """Find all PrimaryNode instances matching the given name."""
        matches: List[PrimaryNode] = []

        if self._matches_name(name, with_alias):
            matches.append(self)

        for child in self.children:
            matches.extend(child.find_all_primary(name, with_alias))

        return matches

    def _find_primary(self, name, with_alias=False) -> "PrimaryNode":
        """Return the first PrimaryNode from the current subtree with the given name."""

        matches = self.find_all_primary(name, with_alias)

        if not matches:
            raise NodeNotFoundError(name)

        # If we're matching with alias, multiple children can produce a match.
        # We therefore select the shortest match
        matches.sort(key=lambda n: len(n.precursors))

        return matches[0]

    def find_primary(
        self, name_or_path: Union[str, Iterable[str]], with_alias=False
    ) -> "PrimaryNode":
        """Find a primary node by name or path."""
        if isinstance(name_or_path, str):
            name_or_path = (name_or_path,)

        anchor = self
        while name_or_path:
            head, *name_or_path = name_or_path
            anchor = anchor._find_primary(head, with_alias)

        return anchor

    def find_tag(self, name_or_path: Union[str, Iterable[str]]) -> TagNode:
        """Find specified tag in this node or its parents."""
        for tag in self.tags:
            try:
                return tag.find_tag(name_or_path)
            except NodeNotFoundError:
                pass

        if self.parent is not None:
            return self.parent.find_tag(name_or_path)

        raise NodeNotFoundError(name_or_path)

    def get_applicable_virtuals(self):
        yield from self.virtuals

        if self.parent is not None:
            yield from self.parent.get_applicable_virtuals()

    def find_virtual(self, name_or_path: Union[str, Iterable[str]]) -> VirtualNode:
        """Find specified virtual node in this node or its parents."""
        if isinstance(name_or_path, str):
            name_or_path = (name_or_path,)
        else:
            name_or_path = tuple(name_or_path)

        anchor = self
        while len(name_or_path) > 1:
            head, *name_or_path = name_or_path
            anchor = anchor._find_primary(head)

        name = name_or_path[0]  # type: ignore

        for virtual in self.get_applicable_virtuals():
            if virtual.name.casefold() == name.casefold():
                return virtual

        raise NodeNotFoundError(name)

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
        self, name_or_path: Union[str, Iterable[str]], with_alias=False
    ) -> Union["PrimaryNode", "TagNode"]:
        """Find a real node (PrimaryNode or TagNode) by name or path."""
        try:
            return self.find_primary(name_or_path, with_alias)
        except NodeNotFoundError:
            pass

        return self.find_tag(name_or_path)

    def get_description(
        self,
        descriptors: Iterable[Union[str, Iterable[str]]],
        ignore_missing_intermediaries=False,
        with_alias=False,
        on_conflict: Literal["replace", "raise"] = "replace",
    ) -> "Description":
        """
        Get a PolyDescription from a list of descriptors.

        Args:
            descriptors (Iterable[Union[str, Iterable[str]]]): The descriptors to parse.
            ignore_missing_intermediaries (bool, optional): Whether to ignore missing intermediaries. Defaults to False.
            with_alias (bool, optional): Whether to consider aliases. Defaults to False.
            on_conflict (Literal["replace", "raise"], optional): Conflict resolution strategy. Defaults to "replace".

        Returns:
            PolyDescription: The parsed PolyDescription.
        """
        # Turn into tuple
        descriptors = tuple(descriptors)

        def process_descriptors(
            description: "Description", descriptors: Sequence, with_alias
        ):
            unmatched_parts = []

            # Requires descriptors to be a sequence
            while descriptors:
                head, *descriptors = descriptors

                if head and head[0] == "!":
                    negate = True
                    head = head[1:]
                else:
                    negate = False

                try:
                    node = description.anchor.find_real_node(head, with_alias)
                except NodeNotFoundError:
                    pass
                else:
                    if not unmatched_parts or isinstance(node, PrimaryNode):
                        description.add(node.negate(negate), on_conflict=on_conflict)

                        if unmatched_parts and not ignore_missing_intermediaries:
                            raise ValueError(
                                f"Unmatched parts {unmatched_parts} before current {head}"
                            )

                        # Reset unmatched parts
                        unmatched_parts.clear()
                        continue
                    # else:
                    # If a TagNode is found but there are unmatched_parts: Treat as not found

                try:
                    virtual = description.anchor.find_virtual(head)
                except NodeNotFoundError:
                    pass
                else:
                    description.add(virtual.description, on_conflict=on_conflict)

                    if unmatched_parts and not ignore_missing_intermediaries:
                        raise ValueError(f"Unmatched parts: {unmatched_parts}")

                    # Reset unmatched parts
                    unmatched_parts.clear()
                    continue

                unmatched_parts.append(head)
            return description, unmatched_parts

        description, unmatched_parts = process_descriptors(
            Description(self), descriptors, False
        )

        if unmatched_parts and with_alias:
            description, unmatched_parts = process_descriptors(
                description, unmatched_parts, True
            )

        if unmatched_parts:
            raise ValueError(f"Unmatched suffix: {unmatched_parts}")

        return description

    def parse_description(
        self,
        description: str,
        ignore_missing_intermediaries=False,
        with_alias=False,
        on_conflict: Literal["replace", "raise"] = "replace",
    ):
        """
        Parse a description string into a PolyDescription.

        Args:
            description (str): The description string to parse.
            ignore_missing_intermediaries (bool, optional): Whether to ignore missing intermediaries. Defaults to False.
            with_alias (bool, optional): Whether to consider aliases. Defaults to False.
            on_conflict (Literal["replace", "raise"], optional): Conflict resolution strategy. Defaults to "replace".

        Returns:
            PolyDescription: The parsed PolyDescription.
        """
        descriptors = _tokenize_expression_str(description)
        return self.get_description(
            descriptors, ignore_missing_intermediaries, with_alias, on_conflict
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
        if isinstance(other, PrimaryNode):
            return self in other.precursors

        if isinstance(other, TagNode):
            return False

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

    @property
    def descriptors(self) -> Sequence[Descriptor]:
        """Return all descriptors including the anchor and qualifiers."""
        return [self.anchor] + self.qualifiers

    def to_binary_raw(self) -> Mapping["RealNode", bool]:
        """Convert the description to a binary representation with nodes and their active state."""
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
            # anchor is empty or more general than other: replace
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
        if other.primary_parent is not None:
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

    def _remove_poly_description(self, other: "Description"):
        for descriptor in other.descriptors:
            self.remove(descriptor)

    def _remove_primary(self, other: "PrimaryNode"):
        if other <= self.anchor:
            # Delete the precluded portion of the anchor
            if other.parent is None:
                raise ValueError("Cannot remove root")

            self.anchor = other.parent

    def _remove_qualifier(self, other: Descriptor):
        # Replace qualifiers that imply "other" with their nearest ancestor that does not imply "other"

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
            self._remove_poly_description(other)

        elif isinstance(other, PrimaryNode):
            self._remove_primary(other)

        elif isinstance(other, Descriptor):
            self._remove_qualifier(other)

        else:
            raise ValueError(f"Unexpected type of other: {type(other)}")

        return self

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
        # Sort qualifiers alphabetically for stable lookup
        qualifiers = sorted([q.format(anchor=self.anchor) for q in self.qualifiers])

        return shlex.join([str(self.anchor)] + qualifiers)
