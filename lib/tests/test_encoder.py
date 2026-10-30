from hypothesis import example, given
import numpy as np
from polytaxo import Taxonomy
from polytaxo.core import Description, RealNode
from polytaxo.encoder import DescriptionEncoder
from tests.data import taxonomy_dict
from tests.hypotheses import description

taxonomy = Taxonomy.from_dict(taxonomy_dict)


@given(description(taxonomy.root))
@example(taxonomy.parse_description("Copepoda/Calanus sex:male view:lateral"))
@example(taxonomy.parse_description("Copepoda !lateral"))
def test_DescriptionEncoder(description: Description):
    nodes = sorted(taxonomy.root.walk(), key=lambda n: n.path)
    nodes_to_encode = set(nodes)

    description_encoder = DescriptionEncoder(taxonomy, nodes)

    binary_description = description_encoder._description_to_binary(
        description,
        nodes_to_encode,
        negative_propagation_depth=1,
    )

    for node in nodes:
        if node <= description:
            assert binary_description.get(node, 0.5) == 1
        elif ~node in description.qualifiers:
            assert binary_description.get(node, 0.5) == 0.0

    encoded = np.array(
        [binary_description.get(node, 0.5) for node in nodes], dtype="float32"
    )

    node_indexes = {n: i for i, n in enumerate(nodes)}
    rivalling_descendants = description_encoder._get_all_rivalling_descendants(
        nodes_to_encode
    )
    description2 = description_encoder._binary_to_description(
        encoded, node_indexes, rivalling_descendants
    )

    assert description == description2

    for node, score in zip(nodes, encoded):
        if score == 1.0:
            assert node <= description
        elif score == 0.0:
            assert ~node <= description


# def test_poly_taxonomy_binary():
#     # A concrete example of a Taxonomy
#     taxonomy = Taxonomy.from_dict(taxonomy_dict)
#     taxonomy.print_tree()

#     encoder = DescriptionEncoder(
#         taxonomy, [n for n in taxonomy.root.walk() if isinstance(n, RealNode)]
#     )

#     # Get a certain description
#     Calanus_male_lateral = taxonomy.parse_description(
#         "Copepoda/Calanus sex:male view:lateral"
#     )

#     encoder._description_to_binary(Calanus_male_lateral)

#     assert Calanus_male_lateral.to_binary_str() == {
#         "/Copepoda": True,
#         "/Copepoda/Calanus": True,
#         "/Copepoda/Metridia": False,
#         "/Copepoda/Other": False,
#         "/Copepoda/view:lateral": True,
#         "/Copepoda/view:frontal": False,
#         "/Copepoda/view:ventral": False,
#         "/Copepoda/sex:male": True,
#         "/Copepoda/sex:female": False,
#     }

#     Copepoda_not_lateral = taxonomy.parse_description("Copepoda !lateral")
#     assert Copepoda_not_lateral.to_binary_str() == {
#         "/Copepoda": True,
#         "/Copepoda/view:lateral": False,
#         # TODO: Is this what we want? (`"... view:lateral": False` would be enough)
#         "/Copepoda/view:lateral:left": False,
#         "/Copepoda/view:lateral:right": False,
#     }

#     # Check roundtripping of binary description
#     Calanus_male_lateral_multilabel = Calanus_male_lateral.to_multilabel(fill_na=0.5)
#     assert Calanus_male_lateral == taxonomy.parse_probabilities(
#         Calanus_male_lateral_multilabel
#     )

#     Copepoda_female = taxonomy.parse_description("Copepoda sex:female").qualifiers[0]
#     assert isinstance(Copepoda_female, TagNode) and Copepoda_female.name == "female"
#     assert Copepoda_female.index is not None

#     Copepoda_male = taxonomy.parse_description("Copepoda sex:male").qualifiers[0]
#     assert isinstance(Copepoda_male, TagNode) and Copepoda_male.name == "male"
#     assert Copepoda_male.index is not None

#     # Check that max score is used
#     Calanus_male_lateral_multilabel = Calanus_male_lateral.to_multilabel(fill_na=0.5)
#     # Set score for female high, but male is higher
#     Calanus_male_lateral_multilabel[Copepoda_female.index] = 0.95
#     assert (
#         Calanus_male_lateral_multilabel[Copepoda_female.index]
#         < Calanus_male_lateral_multilabel[Copepoda_male.index]
#     )
#     assert Calanus_male_lateral == taxonomy.parse_probabilities(
#         Calanus_male_lateral_multilabel, thr_pos_rel=0.0
#     )

#     # Check that max score is used
#     Calanus_male_lateral_multilabel = Calanus_male_lateral.to_multilabel(fill_na=0.5)
#     # Set score for female high, but male is higher
#     Calanus_male_lateral_multilabel[Copepoda_female.index] = 0.95
#     assert (
#         Calanus_male_lateral_multilabel[Copepoda_female.index]
#         < Calanus_male_lateral_multilabel[Copepoda_male.index]
#     )
#     assert Calanus_male_lateral == taxonomy.parse_probabilities(
#         Calanus_male_lateral_multilabel, thr_pos_rel=0.0
#     )

#     # Check that relative threshold prevents descending into the hierarchy
#     Calanus_male_lateral_multilabel = Calanus_male_lateral.to_multilabel(fill_na=0.5)
#     # Set score for female high, but male is higher
#     Calanus_male_lateral_multilabel[Copepoda_female.index] = 0.95
#     assert (
#         Calanus_male_lateral_multilabel[Copepoda_female.index]
#         < Calanus_male_lateral_multilabel[Copepoda_male.index]
#     )
#     Calanus_lateral = taxonomy.parse_description("Calanus view:lateral")
#     assert Calanus_lateral == taxonomy.parse_probabilities(
#         Calanus_male_lateral_multilabel, thr_pos_rel=0.5
#     )
