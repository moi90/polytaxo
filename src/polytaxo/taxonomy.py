import re
from typing import (
    Any,
    Iterable,
    List,
    Literal,
    Mapping,
    Sequence,
    Tuple,
    Union,
    overload,
)

from .descriptor import Descriptor
from .parser import tokenize
from .core import (
    IndexProvider,
    Description,
    ClassNode,
    RealNode,
    TagNode,
    TOnConflictLiteral,
    fill_in_doc,
)
from .core import _doc_fields as _core_doc_fields


class Expression:
    """
    A class representing an expression for matching and modifying Description objects.

    Args:
        include (Description): A description to add / include in matches.
        exclude (List[Description]): A list of descriptions to remove / exclude from matches.
    """

    def __init__(
        self,
        include: Description,
        exclude: Sequence[Union[Description, Descriptor]],
    ):
        self.include = include
        self.exclude = exclude

    @fill_in_doc(_core_doc_fields)
    @staticmethod
    def from_string(
        anchor: ClassNode,
        expression_str: str,
        with_alias=False,
        on_conflict: TOnConflictLiteral = "replace",
    ) -> "Expression":
        """
        Parse an expression string into an `Expression` object.

        Args:
            {anchor_arg}
            expression_str (str): The expression string containing including and excluding descriptors.
            with_alias (bool, optional): Whether to consider aliases in matching nodes.
                Defaults to False.
            on_conflict ('replace', 'raise', or 'skip', optional): Strategy for handling conflicts.
                Defaults to "replace".

        Returns:
            Expression: An `Expression` object with the parsed include and exclude descriptors.

        Raises:
            ValueError: If an unexpected token or state is encountered in the expression string.
        """

        include = Description(anchor)
        exclude: List[Descriptor] = []

        tokens = iter(tokenize(expression_str))

        NEUTRAL = 0
        IN_INCLUDED_PARENTHESIS = 1
        EXCLUDE = 2
        IN_EXCLUDED_PARENTHESIS = 3

        state = NEUTRAL
        description_tokens = []
        while True:
            token = next(tokens, None)

            if state == NEUTRAL:
                if token == "(":
                    state = IN_INCLUDED_PARENTHESIS
                    continue
                elif isinstance(token, tuple) or token == "!":
                    description_tokens.append(token)
                    continue
                elif token is None:
                    # Flush current description
                    if description_tokens:
                        include._parse_description_tokens(
                            iter(description_tokens),
                            with_alias=with_alias,
                            on_conflict=on_conflict,
                        )
                        description_tokens = []
                    break
                elif token == "-":
                    # Flush currently saved description
                    if description_tokens:
                        include._parse_description_tokens(
                            iter(description_tokens),
                            with_alias=with_alias,
                            on_conflict=on_conflict,
                        )
                        description_tokens = []
                    state = EXCLUDE
                    continue
                else:
                    raise ValueError(f"Unexpected token: {token} (state={state})")
            elif state == EXCLUDE:
                if token == "(":
                    state = IN_EXCLUDED_PARENTHESIS
                    continue
                elif token == "!":
                    description_tokens.append(token)
                    continue
                elif isinstance(token, tuple):
                    description_tokens.append(token)
                    # Flush current description: Extend `exclude` with individual descriptors
                    exclude.extend(
                        Description(include.anchor)._parse_description_tokens(
                            iter(description_tokens),
                            with_alias=with_alias,
                            on_conflict=on_conflict,
                        )
                    )
                    description_tokens = []
                    state = NEUTRAL
                    continue
                else:
                    raise ValueError(f"Unexpected token: {token} (state={state})")
            elif state == IN_INCLUDED_PARENTHESIS:
                if token == ")":
                    raise NotImplementedError()
                    state = NEUTRAL
                    continue
                else:
                    raise ValueError(f"Unexpected token: {token}")
            elif state == IN_EXCLUDED_PARENTHESIS:
                if isinstance(token, tuple) or token == "!":
                    description_tokens.append(token)
                elif token == ")":
                    # Flush
                    if description_tokens:
                        # Append complete description to `exclude`
                        d = Description(include.anchor)
                        d._parse_description_tokens(
                            iter(description_tokens),
                            with_alias=with_alias,
                            on_conflict=on_conflict,
                        )
                        exclude.append(d)
                        description_tokens = []
                    state = NEUTRAL
                    continue
                else:
                    raise ValueError(f"Unexpected token: {token} (state={state})")
            else:
                raise ValueError(f"Unexpected state: {state}")

        return Expression(include, exclude)

    def match(self, description: Description) -> bool:
        """
        Check if a given Description matches the expression.

        A matches if A <= description
        !A matches if !A <= description
        -A matches if not (A <= description)
        -!A matches if not (!A <= description)

        Args:
            description (Description): The description to match.

        Returns:
            bool: True if the description matches, False otherwise.
        """
        if not (self.include <= description):
            return False

        for excl in self.exclude:
            if excl <= description:
                return False

        return True

    def apply(
        self,
        description: Description,
        on_conflict: TOnConflictLiteral = "replace",
    ) -> Description:
        """
        Apply the expression (in-place) to the given Description.

        A/!A: A/!A is added to the description.
        -A/-!A: A/!A is removed from the description.

        Args:
            description (Description): The description to modify.
            on_conflict ("replace", "raise" or "skip", optional): Conflict resolution strategy. Defaults to "replace".

        Returns:
            Description: The modified Description.
        """
        description.add(self.include, on_conflict)

        for excl in self.exclude:
            description.remove(excl)

        return description

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Expression):
            return NotImplemented

        return (self.include == other.include) and (self.exclude == other.exclude)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(include={self.include!r}, exclude={self.exclude!r})>"

    def __str__(self) -> str:
        if not self.exclude:
            return str(self.include)

        exclude = []
        for excl in self.exclude:
            if isinstance(excl, Description):
                exclude.append(f"-({excl.format(self.include.anchor)})")
            elif isinstance(excl, Descriptor):
                exclude.append(f"-{excl.format(self.include.anchor)}")
            else:
                raise ValueError(f"Unexpected exclusion clause {excl!r}")

        return str(self.include) + " " + " ".join(exclude)


class Taxonomy:
    """Taxonomy consisting of class nodes and tag nodes."""

    def __init__(self, root: ClassNode) -> None:
        self.root = root

    @classmethod
    def from_dict(cls, tree_dict: Mapping):
        """Create a PolyTaxonomy from a nested dictionary representation."""

        root = ClassNode.from_dict("", tree_dict, None)

        return cls(root)

    @classmethod
    def from_flat_dict(cls, flat_dict: Mapping[str, Any]):
        """Create a PolyTaxonomy from a flat dictionary representation.

        [/<class-name>]*[:<tag-name>]* => {data}
        """

        token_pattern = re.compile(r"([/:])([^/:]+)")

        def _get_or_add_node(
            path: str,
            node: ClassNode | TagNode,
            name: str,
            is_class: bool,
            data: Mapping[str, Any],
        ) -> ClassNode | TagNode:
            if is_class:
                if not isinstance(node, ClassNode):
                    raise ValueError(
                        f"Invalid flat taxonomy path {path}: class segment cannot follow a tag segment"
                    )
                for child in node.classes:
                    if child.name == name:
                        if data:
                            raise ValueError(
                                f"Duplicate class node {name!r} in flat taxonomy path {path!r}. "
                                "Define parent paths before descendants."
                            )
                        return child
                return node.add_class(ClassNode.from_dict(name, data, node))

            for child in node.tags:
                if child.name == name:
                    if data:
                        raise ValueError(
                            f"Duplicate tag node {name!r} in flat taxonomy path {path!r}. "
                            "Define parent paths before descendants."
                        )
                    return child

            return node.add_tag(TagNode.from_dict(name, data, node))

        root = None

        deferred_virtuals: list[tuple[ClassNode, Any]] = []

        for path, data in flat_dict.items():
            # Make a copy to avoid modifying the original
            data = dict(data)
            virtuals = data.pop("virtuals", None)

            if not isinstance(path, str):
                raise TypeError(f"Expected string key, got {type(path).__name__}")

            # If the entry has an empty path, it is the root node
            if not path:
                if root is not None:
                    raise ValueError("Root node already defined")

                root = ClassNode.from_dict("", data, None)
            else:
                tokens = list(token_pattern.finditer(path))

                if not tokens or "".join(m.group(0) for m in tokens) != path:
                    raise ValueError(
                        "Invalid flat taxonomy path "
                        f"{path!r}. Expected [/<class-name>]*[:<tag-name>]*"
                    )

                # If the root node is not defined yet, create an empty root node
                if root is None:
                    root = ClassNode("", None, None)

                node: ClassNode | TagNode = root

                # Iterate through all tokens except the last one, which is the node to add data to
                for token in tokens[:-1]:
                    separator, name = token.groups()

                    node = _get_or_add_node(path, node, name, separator == "/", {})

                # Add data to the last node
                separator, name = tokens[-1].groups()
                node = _get_or_add_node(path, node, name, separator == "/", data)

            if virtuals is not None:
                deferred_virtuals.append((node, virtuals))

        # Finally, create virtual nodes (which may reference tags and children)
        for node, virtuals in deferred_virtuals:
            node._add_virtuals_from_dict(virtuals)

        return cls(root or ClassNode("", None, None))

    @classmethod
    def from_yaml(cls, yaml_fn):
        """Create a PolyTaxonomy from a YAML file."""
        import yaml

        with open(yaml_fn) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> Mapping:
        """Convert the PolyTaxonomy to a dictionary representation."""
        return self.root.to_dict()

    def to_flat_dict(self) -> Mapping[str, Any]:
        """Convert the PolyTaxonomy to a flat dictionary representation.

        [/<class-name>]*[:<tag-name>]* => {data}
        """
        flat_dict: dict[str, Any] = {}

        for node in self.root.walk():
            node_dict = node.to_dict(exclude_real_children=True)
            if node.real_children and not node_dict:
                continue

            flat_dict[node.format()] = node_dict

        return flat_dict

    def parse_description(
        self,
        description: str,
        with_alias=False,
        on_conflict: TOnConflictLiteral = "replace",
    ) -> Description:
        """
        Parse a description string into a Description.

        Args:
            description (str): The description string to parse.
            with_alias (bool, optional): Whether to consider aliases. Defaults to False.
            on_conflict ("replace", "raise", or "skip", optional): Conflict resolution strategy. Defaults to "replace".

        Returns:
            Description: The parsed Description.
        """

        return Description.from_string(
            self.root,
            description,
            with_alias=with_alias,
            on_conflict=on_conflict,
        )

    def parse_expression(self, expression_str: str) -> Expression:
        """
        Parse an expression string into an `Expression` object.

        Args:
            expression_str (str): The expression string containing including and excluding descriptors.
            with_alias (bool, optional): Whether to consider aliases in matching nodes.
                Defaults to False.
            on_conflict ("replace", "raise", or "skip", optional): Strategy for handling conflicts.
                Defaults to "replace".

        Returns:
            Expression: An `Expression` object with the parsed include and exclude descriptors.

        Raises:
            ValueError: If an unexpected token or state is encountered in the expression string.
        """

        return Expression.from_string(self.root, expression_str)

    @overload
    def parse_lineage(
        self,
        names: Iterable[str],
        *,
        with_alias: bool = False,
        on_conflict: TOnConflictLiteral = "replace",
        ignore_unmatched_intermediaries: bool = False,
        return_unmatched_suffix: Literal[True],
    ) -> Tuple["Description", Tuple[str, ...]]: ...

    @overload
    def parse_lineage(
        self,
        names: Iterable[str],
        *,
        with_alias: bool = False,
        on_conflict: TOnConflictLiteral = "replace",
        ignore_unmatched_intermediaries: bool = False,
        return_unmatched_suffix: Literal[False] = False,
    ) -> "Description": ...

    @fill_in_doc(_core_doc_fields)
    def parse_lineage(
        self,
        names: Iterable[str],
        *,
        with_alias: bool = False,
        on_conflict: TOnConflictLiteral = "replace",
        ignore_unmatched_intermediaries: bool = False,
        return_unmatched_suffix: bool = False,
    ) -> Tuple[Description, Tuple[str, ...]] | Description:
        """
        Parse a sequence of names into a Description.

        Args:
            names (iterable of str): The sequence of names to parse into a Description.
            {with_alias_arg}
            {on_conflict_arg}
            {ignore_unmatched_intermediaries_arg}
            {return_unmatched_suffix_arg}

        Returns:
            Description: The parsed Description.
        """

        return Description.from_lineage(
            self.root,
            names,
            with_alias=with_alias,
            on_conflict=on_conflict,
            ignore_unmatched_intermediaries=ignore_unmatched_intermediaries,
            return_unmatched_suffix=return_unmatched_suffix,
        )

    def fill_indices(self):
        """Fill indices for all nodes in the taxonomy."""
        index_provider = IndexProvider()
        for node in self.root.walk():
            if isinstance(node, RealNode) and node.index is not None:
                index_provider.remove(node.index)

        self.root.fill_indices(index_provider)

        return index_provider.n_labels

    def format_tree(self, extra=None, virtuals=False):
        """Format the taxonomy as a tree."""
        return self.root.format_tree(extra, virtuals)

    def print_tree(self, extra=None, virtuals=False):
        """Print the taxonomy as a tree."""
        print(self.format_tree(extra, virtuals))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Taxonomy):
            return NotImplemented

        return self.root == other.root

    def __str__(self) -> str:
        return self.format_tree()
