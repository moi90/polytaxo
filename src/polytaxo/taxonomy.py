import operator as op
import shlex
from collections import defaultdict
from typing import Iterable, Literal, Mapping, Optional, Sequence, Union

from .descriptor import Descriptor
from .parser import _tokenize_expression_str
from .core import (
    IndexProvider,
    Description,
    PrimaryNode,
    RealNode,
    TagNode,
    TOnConflictLiteral,
)


class Expression:
    """
    A class representing an expression for matching and modifying PolyDescription objects.

    Args:
        include (PolyDescription): A description to add / include in matches.
        exclude (List[PolyDescription]): A list of descriptions to remove / exclude from matches.
    """

    def __init__(
        self,
        include: Description,
        exclude: Sequence[Union[Description, Descriptor]],
    ):
        self.include = include
        self.exclude = exclude

    def match(self, description: Description) -> bool:
        """
        Check if a given PolyDescription matches the expression.

        A matches if A <= description
        !A matches if !A <= description
        -A matches if not (A <= description)
        -!A matches if not (!A <= description)

        Args:
            description (PolyDescription): The description to match.

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
        Apply the expression (in-place) to the given PolyDescription.

        A/!A: A/!A is added to the description.
        -A/-!A: A/!A is removed from the description.

        Args:
            description (PolyDescription): The description to modify.
            on_conflict ("replace", "raise" or "skip", optional): Conflict resolution strategy. Defaults to "replace".

        Returns:
            PolyDescription: The modified PolyDescription.
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
        exclude = []
        for excl in self.exclude:
            if isinstance(excl, Descriptor):
                exclude.append(f"-{excl.format(self.include.anchor)}")
            else:
                exclude.append(f"-({excl})")
        return str(self.include) + " " + shlex.join(exclude)


class PolyTaxonomy:
    """Taxonomy with multiple roots."""

    def __init__(self, root: PrimaryNode) -> None:
        self.root = root

    @classmethod
    def from_dict(cls, tree_dict: Mapping):
        """Create a PolyTaxonomy from a dictionary representation."""
        # Ensure single node
        (key, data), *remainder = list(tree_dict.items())

        if remainder:
            raise ValueError("Only one root node is allowed")

        root = PrimaryNode.from_dict(key, data, None)

        return cls(root)

    def to_dict(self) -> Mapping:
        """Convert the PolyTaxonomy to a dictionary representation."""
        return {self.root.name: self.root.to_dict()}

    def get_description(
        self,
        descriptors: Iterable[Union[str, Iterable[str]]],
        ignore_missing_intermediaries=False,
        with_alias=False,
        on_conflict: Literal["replace", "raise"] = "replace",
    ) -> Description:
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
        return self.root.get_description(
            descriptors, ignore_missing_intermediaries, with_alias, on_conflict
        )

    def parse_description(
        self,
        description: str,
        ignore_missing_intermediaries=False,
        with_alias=False,
        on_conflict: Literal["replace", "raise"] = "replace",
    ) -> Description:
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
        return self.root.parse_description(
            description, ignore_missing_intermediaries, with_alias, on_conflict
        )

    def parse_expression(self, expression_str: str) -> Expression:
        """Parse an expression string into an Expression object."""
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
        """Get a node by name."""
        (path,) = _tokenize_expression_str(node_name)

        return self.root.find_real_node(path)

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
            baseline (Optional[PolyDescription], optional): The baseline description. Defaults to None.
            thr_pos_abs (float, optional): The absolute positive threshold. Defaults to 0.9.
            thr_pos_rel (float, optional): The relative positive threshold. Defaults to 0.25.
            thr_neg (float, optional): The negative threshold. Defaults to 0.1.

        Returns:
            PolyDescription: The resulting description.
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

        def handle_primary(description: Description, node: PrimaryNode) -> Description:
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

    def __eq__(self, other) -> bool:
        if not isinstance(other, PolyTaxonomy):
            return NotImplemented

        return self.root == other.root

    def __str__(self) -> str:
        return self.format_tree()
