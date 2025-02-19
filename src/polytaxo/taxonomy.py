import operator as op
from collections import defaultdict
from typing import (
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
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
    """Taxonomy consisting of primary nodes and tag nodes."""

    def __init__(self, root: ClassNode) -> None:
        self.root = root

    @classmethod
    def from_dict(cls, tree_dict: Mapping):
        """Create a PolyTaxonomy from a dictionary representation."""

        root = ClassNode.from_dict("", tree_dict, None)

        return cls(root)

    @classmethod
    def from_yaml(cls, yaml_fn):
        """Create a PolyTaxonomy from a YAML file."""
        import yaml

        with open(yaml_fn) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> Mapping:
        """Convert the PolyTaxonomy to a dictionary representation."""
        return {self.root.name: self.root.to_dict()}

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
        with_alias: bool,
        on_conflict: TOnConflictLiteral,
        ignore_unmatched_intermediaries: bool,
        return_unmatched_suffix: Literal[True],
    ) -> Tuple["Description", List[str]]: ...

    @overload
    def parse_lineage(
        self,
        names: Iterable[str],
        *,
        with_alias: bool,
        on_conflict: TOnConflictLiteral,
        ignore_unmatched_intermediaries: bool,
        return_unmatched_suffix: Literal[False],
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
    ) -> Tuple[Description, List[str]] | Description:
        """
        Parse a sequence of names into a Description.

        Args:
            names (iterable of str): ...
            {with_alias_arg}
            {on_conflict_arg}
            {ignore_unmatched_intermediaries_arg}

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

    def parse_probabilities(
        self,
        probabilities: Union[Mapping[int, float], Sequence[float]],
        *,
        baseline: Optional[Description] = None,
        thr_pos_abs=0.9,
        thr_pos_rel=0.25,
        thr_neg=0.1,
    ) -> Description:
        """
        Turn per-node probability scores (between 0 and 1) into a description.

        The algorithm proceeds along the hierarchy.
        If the score for a node exceeds the positive thresholds, the node is added to the
        description and the algorithm descends. If the score falls below the negative threshold,
        the negated node is added to the description.

        If a baseline description is supplied, it is updated with compatible predictions.
        Incompatible predictions will not be used, even if they obtain higher scores than any compatible prediction.

        Args:
            probabilities (Union[Mapping[int, float], Sequence[float]]): The probability scores for the nodes.
            baseline (Optional[Description], optional): The baseline description. Defaults to None.
            thr_pos_abs (float, optional): The absolute positive threshold. Defaults to 0.9.
            thr_pos_rel (float, optional): The relative positive threshold. Defaults to 0.25.
            thr_neg (float, optional): The negative threshold. Defaults to 0.1.

        Returns:
            Description: The resulting description.
        """
        if isinstance(probabilities, Mapping):
            probabilities = defaultdict(lambda: 0.5, probabilities)
        else:  # Sequence
            probabilities = defaultdict(
                lambda: 0.5, {i: k for i, k in enumerate(probabilities)}
            )

        if baseline is None:
            baseline = Description(self.root)

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

        def handle_primary(description: Description, node: ClassNode) -> Description:
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
        primary_node: ClassNode
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

    def __eq__(self, other) -> bool:
        if not isinstance(other, Taxonomy):
            return NotImplemented

        return self.root == other.root

    def __str__(self) -> str:
        return self.format_tree()
