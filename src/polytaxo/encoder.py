from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Set

import numpy as np

from .core import ClassNode, Description, NegatedRealNode, RealNode, TRealNode, TagNode
from .taxonomy import Taxonomy
import operator as op

NAN = float("nan")


class DescriptionEncoder:
    """
    Encode descriptions into a matrix of binary values.

    Args:
        taxonomy: The taxonomy to use.

    .. note::
        There are different encoding strategies.
        The simplest one is to encode each descriptor independently any other node's annotation status.
        This, however, can lead to very imbalanced data, as the number of positive annotations is usually much smaller.
        Originally, positive annotations would propagate towards the root, deactivating sibling nodes,
        and negative annotations would propagate towards the leaves.
        It should be tested whether negative annotations should really be propagated or if children should be skipped.
    """

    def __init__(self, taxonomy: Taxonomy, nodes: Optional[List[RealNode]] = None):
        self.taxonomy = taxonomy

        self.nodes: Optional[List[RealNode]] = nodes

    def fit(
        self,
        descriptions: Sequence[Description],
        include_nodes: Optional[Set[RealNode]] = None,
        exclude_nodes: Optional[Set[RealNode]] = None,
    ):
        """
        Fit the encoder to the given descriptions.
        """

        if self.nodes is not None:
            raise ValueError("DescriptionEncoder has already been fitted.")

        if include_nodes is None:
            include_nodes = set()

        if exclude_nodes is None:
            exclude_nodes = set()

        nodes: Set[RealNode] = set()
        nodes.update(include_nodes)
        for description in descriptions:
            for node in description.descriptors:
                if isinstance(node, NegatedRealNode):
                    node = ~node
                if not isinstance(node, RealNode):
                    raise ValueError(f"Invalid descriptor: {node}")

                if node in exclude_nodes:
                    continue

                nodes.add(node)

        self.nodes = sorted(nodes, key=lambda n: n.path)

        return nodes

    def fit_transform(
        self,
        descriptions: Sequence[Description],
        include_nodes: Optional[Set[RealNode]] = None,
        exclude_nodes: Optional[Set[RealNode]] = None,
        negative_propagation_depth: int = 1,
    ):
        """
        Fit the encoder to the given descriptions and return the encoded descriptions.
        """

        # First, fit.
        self.fit(descriptions, include_nodes, exclude_nodes)

        # Then, transform
        return self.transform(descriptions, negative_propagation_depth)

    def _description_to_binary(
        self,
        description: Description,
        nodes_to_encode: Set["RealNode"],
        negative_propagation_depth: int = 1,
    ) -> Dict[RealNode, float]:
        """Convert the description to a binary representation of nodes and their active state."""

        map: Dict[RealNode, float] = {}

        def handle_negative(node: RealNode, max_depth: int):
            # Deactivate node and all successors (up to max_depth)
            if max_depth <= 0:
                return

            if node in nodes_to_encode:
                map[node] = 0.0
                for child in node.real_children:
                    handle_negative(child, max_depth=max_depth - 1)
            else:
                # If the node is not in the set, we don't need to deactivate it
                # but we still need to deactivate its children.
                # The depth is left unchanged.
                for child in node.real_children:
                    handle_negative(child, max_depth=max_depth)

        def handle_positive(node: RealNode):
            # Activate node and all precursors
            for precursor in node.precursors:
                if not isinstance(precursor, node.__class__):
                    continue

                if precursor in nodes_to_encode:
                    map[precursor] = 1.0

                # Deactivate all siblings
                for sibling in precursor.siblings:
                    handle_negative(sibling, max_depth=negative_propagation_depth)

        handle_positive(description.anchor)

        for qualifier in description.qualifiers:
            if isinstance(qualifier, RealNode):
                handle_positive(qualifier)
            elif isinstance(qualifier, NegatedRealNode):
                handle_negative(qualifier.node, max_depth=negative_propagation_depth)
            else:
                raise ValueError(f"Unknown qualifier type: {type(qualifier)}")

        return map

    def transform(
        self,
        descriptions: Sequence[Description],
        negative_propagation_depth: int = 1,
    ):
        """
        Transform the given descriptions into an encoded form using the fitted descriptors.
        """

        if self.nodes is None:
            raise ValueError("DescriptionEncoder has not been fitted.")

        nodes_to_encode = set(self.nodes)

        encoded = np.empty((len(descriptions), len(self.nodes)), dtype="float32")
        for i, description in enumerate(descriptions):
            binary_description = self._description_to_binary(
                description,
                nodes_to_encode,
                negative_propagation_depth=negative_propagation_depth,
            )
            for j, node in enumerate(self.nodes):
                encoded[i, j] = binary_description.get(node, NAN)

        return encoded

    def _get_rivalling_descendants_for(
        self, node: TRealNode, nodes_to_encode
    ) -> Iterator[TRealNode]:
        for c in node.real_children:
            if not isinstance(c, node.__class__):
                continue

            if c in nodes_to_encode:
                yield c
            else:
                yield from self._get_rivalling_descendants_for(c, nodes_to_encode)

    def _get_all_rivalling_descendants(self, nodes_to_encode):
        stack: List[RealNode] = [self.taxonomy.root]
        rivalling_descendants = {}
        while stack:
            node = stack.pop(0)
            rivalling_descendants_for_node = list(
                self._get_rivalling_descendants_for(node, nodes_to_encode)
            )
            rivalling_descendants[node] = rivalling_descendants_for_node
            stack.extend(node.real_children)

        return rivalling_descendants

    def _binary_to_description(
        self,
        probabilities: np.ndarray,
        node_indexes: Mapping[RealNode, int],
        rivalling_descendants: Mapping[RealNode, List[RealNode]],
        baseline: Optional[Description] = None,
        thr_abs=0.9,
        thr_rel=0.25,
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
            thr_abs (float, optional): The absolute threshold. Defaults to 0.9.
            thr_rel (float, optional): The relative threshold. Defaults to 0.25.

        Returns:
            Description: The resulting description.
        """

        if baseline is None:
            description = Description(self.taxonomy.root)
        else:
            description = baseline.copy()

        thr_neg = 1 - thr_abs

        def handle_node(node: RealNode, positives=True):
            # Gather all rival direct (and, in case of index==None, indirect) descendants of tag
            candidate_scores = [
                (n, probabilities[node_indexes[n]]) for n in rivalling_descendants[node]
            ]

            # We need at least one candidate
            if not candidate_scores:
                return

            # Find winner
            candidate_scores.sort(key=op.itemgetter(1))
            winner, winner_score = candidate_scores[-1]

            # Check if winner is good enough
            good_enough = winner_score >= thr_abs

            # If there are other candidates, apply thr_pos_rel
            if good_enough and len(candidate_scores) > 1:
                _, second_score = candidate_scores[-2]
                good_enough &= winner_score - second_score >= thr_rel

            # If there is a good enough winner, store and descend
            if good_enough and positives:
                description.add(winner, on_conflict="skip")
                handle_node(winner)
            else:
                # Otherwise, at least store all negatives
                for loser, score in candidate_scores:
                    if score <= thr_neg:
                        description.add(loser.negate(), on_conflict="skip")
                    else:
                        handle_node(loser, positives=False)

        handle_node(description.anchor)

        # Once the anchor is predicted, predict additional tags
        class_node: ClassNode
        for class_node in description.anchor.precursors:  # type: ignore
            # Tags below a ClassNode are not in rivalry
            for tag in class_node.tags:
                if tag in node_indexes:
                    score = probabilities[node_indexes[tag]]
                    if score > thr_abs:
                        description.add(tag, on_conflict="skip")
                        handle_node(tag)
                    elif score < thr_neg:
                        description.add(tag.negate(), on_conflict="skip")
                    else:
                        handle_node(tag, positives=False)

        return description

    def inverse_transform(
        self, encoded_descriptions: np.ndarray
    ) -> Sequence[Description]:
        """
        Inverse-transform the encoded descriptions into the original descriptions.
        """

        if self.nodes is None:
            raise ValueError("DescriptionEncoder has not been fitted.")

        descriptions = []
        for row in encoded_descriptions:
            descriptions.append(self._binary_to_description(row))

        return descriptions
