import abc
import fnmatch
import functools
import itertools
import operator as op
import re
import shlex
from collections import defaultdict
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


class NodeNotFoundError(Exception):
    pass


class ConflictError(Exception):
    pass


T = TypeVar("T")
TOnConflictLiteral = Literal["replace", "raise", "skip"]


class Descriptor(abc.ABC):
    @property
    @abc.abstractmethod
    def unique_id(self):  # pragma: no cover
        ...

    def __hash__(self) -> int:
        return hash(self.unique_id)

    def __invert__(self):
        return self.negate()

    @abc.abstractmethod
    def negate(self, negate: bool = True) -> "Descriptor":  # pragma: no cover
        ...

    @abc.abstractmethod
    def format(self, anchor: Optional["PrimaryNode"] = None) -> str: ...

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.format()}>"

    def __eq__(self, other: object) -> bool:
        if type(self) != type(other):
            return False

        return self.unique_id == other.unique_id  # type: ignore

    def __le__(self, other) -> bool:  # pragma: no cover
        """Return True if self is implied by other."""
        if isinstance(other, PolyDescription):
            for other_descriptor in other.descriptors:
                if self <= other_descriptor:
                    return True
            return False

        return NotImplemented


class NeverDescriptor(Descriptor):
    """A descriptor that is never implied by another (even itself)."""

    @property
    def unique_id(self):
        raise NotImplementedError()

    def __le__(self, other) -> bool:
        return False

    def negate(self, negate: bool = True) -> Descriptor:
        raise NotImplementedError()

    def format(self, anchor: Optional["PrimaryNode"] = None) -> str:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return object.__repr__(self)


class PolyDescription:
    def __init__(
        self, anchor: "PrimaryNode", qualifiers: Optional[Iterable[Descriptor]] = None
    ) -> None:
        self.anchor: "PrimaryNode" = anchor
        self.qualifiers: List[Descriptor] = []

        if qualifiers is not None:
            for descriptor in qualifiers:
                self.add(descriptor, on_conflict="raise")

    @property
    def descriptors(self) -> Sequence[Descriptor]:
        return [self.anchor] + self.qualifiers

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PolyDescription):
            return False

        return set(self.descriptors) == set(other.descriptors)

    def __hash__(self) -> int:
        return hash(frozenset(self.descriptors))

    def __le__(self, other) -> bool:
        if not isinstance(other, PolyDescription):
            return NotImplemented

        for d in self.descriptors:
            if not (d <= other):
                return False

        return True

    def __str__(self) -> str:
        parts = []
        parts.append(str(self.anchor))

        for node in self.qualifiers:
            parts.append(node.format(anchor=self.anchor))

        return shlex.join(parts)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self!s}>"

    def to_binary_raw(self) -> Mapping["_BaseRealNode", bool]:
        map: Mapping[_BaseRealNode, bool] = {}

        def handle_positive(node: _BaseRealNode):
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

        def handle_negative(node: _BaseRealNode):
            # Deactivate node and all successors
            for successor in node.walk():
                if not isinstance(successor, node.__class__):
                    continue

                if successor._include_in_binary():
                    map[successor] = False

        handle_positive(self.anchor)

        for qualifier in self.qualifiers:
            if isinstance(qualifier, _BaseRealNode):
                handle_positive(qualifier)
            elif isinstance(qualifier, NegatedRealNode):
                handle_negative(qualifier.node)
            else:
                raise ValueError(f"Unknown qualifier type: {type(qualifier)}")

        return map

    def to_binary_str(self) -> Mapping[str, bool]:
        return {str(k): v for k, v in self.to_binary_raw().items()}

    def to_multilabel(self, n_labels=None, fill_na: Any = -1) -> List:
        """Transform the description to a list of target values."""
        int_map = {
            n.index: v for n, v in self.to_binary_raw().items() if n.index is not None
        }

        if n_labels is None:
            n_labels = max(int_map.keys()) + 1

        return [int_map.get(i, fill_na) for i in range(n_labels)]

    def copy(self) -> "PolyDescription":
        return PolyDescription(self.anchor, self.qualifiers)

    def _add_poly_description(
        self,
        other: "PolyDescription",
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
        other: Union["PolyDescription", Descriptor],
        on_conflict: TOnConflictLiteral = "replace",
    ) -> "PolyDescription":
        if on_conflict not in ("replace", "raise", "skip"):
            raise ValueError(f"Unexpected value for on_conflict: {on_conflict}")

        if isinstance(other, PolyDescription):
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

    def _remove_poly_description(self, other: "PolyDescription"):
        for descriptor in other.descriptors:
            self.remove(descriptor)

    def _remove_primary(self, other: "PrimaryNode"):
        if other <= self.anchor:
            # Delete the precluded portion of the anchor
            if other.parent is None:
                raise ValueError("Can not remote root")

            self.anchor = other.parent

    def _remove_qualifier(self, other: "Descriptor"):
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

    def remove(self, other: Union["PolyDescription", Descriptor]):
        if isinstance(other, PolyDescription):
            self._remove_poly_description(other)

        elif isinstance(other, PrimaryNode):
            self._remove_primary(other)

        elif isinstance(other, Descriptor):
            self._remove_qualifier(other)

        else:
            raise ValueError(f"Unexpected type of other: {type(other)}")

        return self


class _BaseNode:
    def __init__(
        self,
        name: str,
        parent: Optional["_BaseNode"],
    ) -> None:
        self.name = name
        self.parent = parent

    def set_parent(self, parent: Optional["_BaseNode"]):
        self.parent = parent
        return self

    @functools.cached_property
    def precursors(self) -> Tuple["_BaseNode", ...]:
        def _(node):
            while node is not None:
                yield node
                node = node.parent

        return tuple(_(self))[::-1]

    @property
    def siblings(self):
        if self.parent is None:
            return tuple()

        if isinstance(self.parent, (PrimaryNode, TagNode)):
            return tuple(c for c in self.parent.children if c is not self)

        return tuple()

    @functools.cached_property
    def path(self):
        return tuple(n.name for n in self.precursors)

    def format(self, anchor: Union["PrimaryNode", None] = None) -> str:
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
    def __init__(self) -> None:
        self.counter = itertools.count()
        self.taken: Set[int] = set()

    def remove(self, index):
        if index in self.taken:
            raise ValueError(f"Index {index} is already taken")

        self.taken.add(index)

    def take(self):
        while True:
            index = next(self.counter)
            if index not in self.taken:
                break

        self.taken.add(index)

        return index

    @property
    def n_labels(self):
        return max(self.taken) + 1


class _BaseRealNode(_BaseNode, Descriptor):
    def __init__(
        self,
        name: str,
        parent: Optional["_BaseNode"],
        index: Optional[int],
    ) -> None:
        super().__init__(name, parent)
        self.index = index

    def to_dict(self) -> Dict[str, Any]:
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
        # A node should be included in the binary output if it is a proper inner node (not a (pseudo-)root) or a leaf
        return (type(self.parent) == type(self)) or not self._get_all_children()

    def fill_indices(self, index_provider: IndexProvider):
        if self.index is None and self._include_in_binary():
            self.index = index_provider.take()
        for child in self._get_all_children():
            if isinstance(child, _BaseRealNode):
                child.fill_indices(index_provider)

    def walk(self) -> Iterator["_BaseRealNode"]:
        yield self
        for child in self._get_all_children():
            if isinstance(child, _BaseRealNode):
                yield from child.walk()

    def negate(self, negate: bool = True) -> "Descriptor":
        if negate:
            return NegatedRealNode(self)
        return self


TRealNode = TypeVar("TRealNode", bound=_BaseRealNode)


class NegatedRealNode(Descriptor, Generic[TRealNode]):
    def __init__(self, node: TRealNode) -> None:
        # We can not negate the root
        if node.parent is None:
            raise ValueError(f"Root node can not be negated")

        self.node = node

    def format(self, anchor: Union["PrimaryNode", None] = None) -> str:
        return f"!{self.node.format(anchor)}"

    @property
    def unique_id(self):
        return ("!",) + self.node.path

    def negate(self, negate: bool = True) -> "Descriptor":
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

        return Descriptor.__le__(self, other)


class TagNode(_BaseRealNode):
    def __init__(
        self,
        name: str,
        parent: Optional["_BaseNode"],
        index: Optional[int],
    ) -> None:
        super().__init__(name, parent, index)

        self.children: List["TagNode"] = []

    @staticmethod
    def from_dict(
        name, data: Optional[Mapping], parent: Union["PrimaryNode", "TagNode"]
    ) -> "TagNode":
        if data is None:
            data = {}

        tag_node = TagNode(name, parent, data.get("index", None))

        for child_name, child_data in data.get("children", {}).items():
            tag_node.add_child(TagNode.from_dict(child_name, child_data, tag_node))

        return tag_node

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        if self.children:
            d["children"] = {c.name: c.to_dict() for c in self.children}

        return d

    def add_child(self, node: "TagNode"):
        self.children.append(node)

    def _get_all_children(self):
        return self.children

    @functools.cached_property
    def primary_parent(self) -> Optional["PrimaryNode"]:
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
        if isinstance(name_or_path, str):
            name_or_path = (name_or_path,)

        node = self
        while name_or_path:
            head, *name_or_path = name_or_path
            node = node._find_tag(head)

        return node

    def format_tree(self, extra_info=None) -> str:
        name = self.name

        extra = []
        if self.index is not None:
            extra.append(f"index={self.index}")

        if extra:
            name = name + " (" + (", ".join(extra)) + ")"

        if (self.index is not None) and (extra_info is not None):
            name += f" [{extra_info[self.index]}]"

        lines = [f"{name}:"]
        for child in self.children:
            lines.append(indent(child.format_tree(extra_info), "  "))

        return "\n".join(lines)

    def __le__(self, other) -> bool:
        if isinstance(other, TagNode):
            return self in other.precursors

        if isinstance(other, PrimaryNode):
            return False

        if isinstance(other, NegatedRealNode):
            return False

        return Descriptor.__le__(self, other)

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


def _tokenize_expression_str(query_str: str):
    # Split into parts, then at : and / and separate off !
    return [
        tuple(filter(None, (re.split("/|:|(!)|(-)", part))))
        for part in shlex.split(query_str)
    ]


class VirtualNode(_BaseNode):
    def __init__(
        self,
        name: str,
        parent: Optional["_BaseNode"],
        description: PolyDescription,
    ) -> None:
        super().__init__(name, parent)

        self.description = description

    def format_tree(self) -> str:
        return f"{self.name} -> {self.description!s}"

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.format_tree()}>"


class PrimaryNode(_BaseRealNode):
    parent: Optional["PrimaryNode"]

    def __init__(
        self,
        name: str,
        parent: Optional["PrimaryNode"],
        index: Optional[int],
        alias: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(name, parent, index)
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
        if data is None:
            data = {}

        # Create node
        node = PrimaryNode(name, parent, data.get("index"), data.get("alias"))

        # Create tags
        for tag_name, tag_data in data.get("tags", {}).items():
            node.add_tag(TagNode.from_dict(tag_name, tag_data, node))

        # Create children (which may reference tags)
        for child_name, child_data in data.get("children", {}).items():
            node.add_child(PrimaryNode.from_dict(child_name, child_data, node))

        # Finally, create virtual nodes (which may reference tags and children)
        for virtual_name, virtual_description in data.get("virtual", {}).items():
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
        d = super().to_dict()
        if self.alias:
            if isinstance(self.alias, str) or len(self.alias) > 1:
                d["alias"] = self.alias
            else:
                d["alias"] = self.alias[0]

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
        self.children.append(node)
        return node

    def add_tag(self, node: TagNode):
        self.tags.append(node)
        return node

    def add_virtual(self, node: VirtualNode):
        self.virtuals.append(node)
        return node

    def _matches_name(self, name: str, with_alias: bool):
        if name == self.name:
            return True

        if with_alias:
            for alias in self.alias:
                if fnmatch.fnmatch(name, alias):
                    return True

        return False

    def find_all_primary(self, name: str, with_alias=False) -> List["PrimaryNode"]:
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

        name = name_or_path[0]

        for virtual in self.virtuals:
            if virtual.name.casefold() == name.casefold():
                return virtual

        if anchor.parent is not None:
            return anchor.parent.find_virtual(name)

        raise NodeNotFoundError(name)

    def format_tree(self, extra=None, virtuals=False) -> str:
        name = self.name
        if self.index is not None:
            name += f" (index={self.index})"
            if extra is not None:
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
    ) -> PolyDescription:
        """
        Takes the path to a node in a monohierarchy and transforms it
        into a polyhierarchical description.
        """

        # Turn into tuple
        descriptors = tuple(descriptors)

        def process_descriptors(
            description: PolyDescription, descriptors: Sequence, with_alias
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
            PolyDescription(self), descriptors, False
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
        descriptors = _tokenize_expression_str(description)
        return self.get_description(
            descriptors, ignore_missing_intermediaries, with_alias, on_conflict
        )

    def union(self, other: "PrimaryNode"):
        if self in other.precursors:
            return other

        if other in self.precursors:
            return self

        raise ValueError(f"{self} and {other} are incompatible")

    def __le__(self, other) -> bool:
        if isinstance(other, PrimaryNode):
            return self in other.precursors

        if isinstance(other, TagNode):
            return False

        if isinstance(other, NegatedRealNode):
            return False

        return Descriptor.__le__(self, other)

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

    def negate(self, negate: bool = True) -> "Descriptor":
        if negate and self.parent is None:
            return NeverDescriptor()

        return super().negate(negate)


class Expression:
    """
    A class representing an expression for matching and modifying PolyDescription objects.

    Args:
        include (PolyDescription): A description to add / include in matches.
        exclude (List[PolyDescription]): A list of descriptions to remove / exclude from matches.
    """

    def __init__(
        self,
        include: PolyDescription,
        exclude: Sequence[Union[PolyDescription, Descriptor]],
    ):
        self.include = include
        self.exclude = exclude

    def match(self, description: PolyDescription) -> bool:
        """
        Check if a given PolyDescription matches the expression.

        A matches if A <= description
        !A matches if !A <= description
        -A matches if not (A <= description)
        -!A matches if not (!A <= description)
        """

        if not (self.include <= description):
            return False

        for excl in self.exclude:
            if excl <= description:
                return False

        return True

    def apply(
        self,
        description: PolyDescription,
        on_conflict: TOnConflictLiteral = "replace",
    ) -> PolyDescription:
        """
        Apply the expression (in-place) to the given PolyDescription.

        A/!A: A/!A is added to the description.
        -A/-!A: A/!A is removed from the description.
        """

        description.add(self.include, on_conflict)

        for excl in self.exclude:
            description.remove(excl)

        return description

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(include={self.include!r}, exclude={self.exclude!r})>"

    def __str__(self) -> str:
        exclude = []
        for excl in self.exclude:
            if isinstance(excl, Descriptor):
                exclude.append(f"-{excl.format(self.include.anchor)}")
            else:
                exclude.append(f"-({excl})")
        return str(self.include) + " " + shlex.join(exclude)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Expression):
            return NotImplemented

        return (self.include == other.include) and (self.exclude == other.exclude)


class PolyTaxonomy:
    """Taxonomy with multiple roots."""

    def __init__(self, root: PrimaryNode) -> None:
        self.root = root

    @classmethod
    def from_dict(cls, tree_dict: Mapping):
        # Ensure single node
        (key, data), *remainder = list(tree_dict.items())

        if remainder:
            raise ValueError("Only one root node is allowed")

        root = PrimaryNode.from_dict(key, data, None)

        return cls(root)

    def to_dict(self) -> Mapping:
        return {self.root.name: self.root.to_dict()}

    def __eq__(self, other) -> bool:
        if not isinstance(other, PolyTaxonomy):
            return NotImplemented

        return self.root == other.root

    def get_description(
        self,
        descriptors: Iterable[Union[str, Iterable[str]]],
        ignore_missing_intermediaries=False,
        with_alias=False,
        on_conflict: Literal["replace", "raise"] = "replace",
    ) -> PolyDescription:
        return self.root.get_description(
            descriptors, ignore_missing_intermediaries, with_alias, on_conflict
        )

    def parse_description(
        self,
        description: str,
        ignore_missing_intermediaries=False,
        with_alias=False,
        on_conflict: Literal["replace", "raise"] = "replace",
    ) -> PolyDescription:
        return self.root.parse_description(
            description, ignore_missing_intermediaries, with_alias, on_conflict
        )

    def parse_expression(self, expression_str: str) -> Expression:
        include = []
        exclude = []
        for part in _tokenize_expression_str(expression_str):
            if part and part[0] == "-":
                exclude.append(part[1:])
            else:
                include.append(part)

        include_dsc = self.get_description(include)

        exclude_descriptors = [
            include_dsc.anchor.get_description([e]).descriptors[-1] for e in exclude
        ]

        return Expression(include_dsc, exclude_descriptors)

    def get_node(self, node_name):
        (path,) = _tokenize_expression_str(node_name)

        return self.root.find_real_node(path)

    def fill_indices(self):
        index_provider = IndexProvider()
        for node in self.root.walk():
            if isinstance(node, _BaseRealNode) and node.index is not None:
                index_provider.remove(node.index)

        self.root.fill_indices(index_provider)

        return index_provider.n_labels

    def format_tree(self, extra=None, virtuals=False):
        return self.root.format_tree(extra, virtuals)

    def print_tree(self, extra=None, virtuals=False):
        print(self.format_tree(extra, virtuals))

    def parse_probabilities(
        self,
        probabilities: Union[Mapping[int, float], Sequence[float]],
        *,
        baseline: Optional[PolyDescription] = None,
        thr_pos_abs=0.9,
        thr_pos_rel=0.25,
        thr_neg=0.1,
    ) -> PolyDescription:
        """
        Turn per-node probability scores (between 0 and 1) into a description.

        The algorithm proceeds along the hierarchy.
        If the score for a node exceeds the positive thresholds, the node is added to the
        description and the algorithm descends. If the score falls below the negative threshold,
        the negated node is added to the description.
        """

        if isinstance(probabilities, Mapping):
            probabilities = defaultdict(lambda: 0.5, probabilities)
        else:  # Sequence
            probabilities = defaultdict(
                lambda: 0.5, {i: k for i, k in enumerate(probabilities)}
            )

        if baseline is None:
            baseline = PolyDescription(self.root)

        description = baseline.copy()

        def handle_tag(tag: TagNode):
            # Gather all rival direct (and, in case of index==None, indirect) descendants of tag
            candidate_scores = [(n, probabilities[n.index]) for n in tag.rivalling_children]  # type: ignore

            # We need at least one candidate
            if not candidate_scores:
                return

            # Find winner
            candidate_scores.sort(key=op.itemgetter(1))
            winner, winner_score = candidate_scores[-1]

            # Check if winner is good enough
            good_enough = winner_score >= thr_pos_abs

            # If there are other candidates, apply thr_pos_rel
            if good_enough and len(candidate_scores) > 1:
                _, second_score = candidate_scores[-2]
                good_enough &= winner_score - second_score >= thr_pos_rel

            # If there is a good enough winner, store and descend
            if good_enough:
                description.add(winner, on_conflict="skip")
                handle_tag(winner)
            else:
                # Otherwise, at least store negatives
                for loser, score in candidate_scores:
                    if score <= thr_neg:
                        description.add(loser.negate(), on_conflict="skip")
                    else:
                        break

        def handle_primary(
            description: PolyDescription, node: PrimaryNode
        ) -> PolyDescription:
            # Gather all rival direct (and, in case of index==None, indirect) descendants of node
            candidate_scores = [(n, probabilities[n.index]) for n in node.rivalling_children]  # type: ignore

            # We need at least one candidate
            if not candidate_scores:
                return description

            # Find winner
            candidate_scores.sort(key=op.itemgetter(1))
            winner, score = candidate_scores[-1]

            # If there is a winner exceeding the threshold, store and descend
            if score >= thr_pos_abs:
                description.add(winner, on_conflict="skip")
                return handle_primary(description, winner)

            # Otherwise, at least store negatives
            for loser, score in candidate_scores:
                if score <= thr_neg:
                    description.add(loser.negate(), on_conflict="skip")
                else:
                    break

            return description

        description = handle_primary(description, description.anchor)

        # Once the anchor is predicted, predict additional tags
        primary_node: PrimaryNode
        for primary_node in description.anchor.precursors:  # type: ignore
            # Tags below a PrimaryNode are not in rivalry
            for tag in primary_node.tags:
                if tag.index is not None:
                    score = probabilities[tag.index]
                    if score <= thr_pos_abs:
                        if score < thr_neg:
                            description.add(tag.negate(), on_conflict="skip")
                        # Continue with next tag, do not descend
                        continue
                    else:  # score > thr_pos:
                        description.add(tag, on_conflict="skip")

                # Descend if index is None or score > thr_pos
                handle_tag(tag)

        return description
