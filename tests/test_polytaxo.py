import pytest

from polytaxo import (
    ConflictError,
    PolyTaxonomy,
    TagNode,
    NodeNotFoundError,
)
from tests.data import taxonomy_dict
from polytaxo.core import Description, NeverDescriptor


def test_poly_taxonomy():
    # A concrete example of a polytaxonomy
    poly_taxonomy = PolyTaxonomy.from_dict(taxonomy_dict)

    assert poly_taxonomy.root.name == ""
    poly_taxonomy.root.find_primary(("", "Copepoda"))
    poly_taxonomy.root.find_real_node(("", "Copepoda"))

    # Test roundtripping
    assert PolyTaxonomy.from_dict(poly_taxonomy.to_dict()) == poly_taxonomy

    # Get a certain description
    Calanus_male_lateral = poly_taxonomy.parse_description(
        "Copepoda/Calanus sex:male view:lateral"
    )

    # Check string representation
    Calanus_male_lateral_str = str(Calanus_male_lateral)
    assert Calanus_male_lateral_str == "/Copepoda/Calanus sex:male view:lateral"

    # Check string parsing
    assert (
        poly_taxonomy.parse_description(Calanus_male_lateral_str)
        == Calanus_male_lateral
    )

    # Check correct anchoring of string representations
    Copepoda_not_Calanus = poly_taxonomy.parse_description("Copepoda !Calanus")
    assert (str(Copepoda_not_Calanus)) == "/Copepoda !Calanus"

    # Assert that missing intermediaries lead to an error
    with pytest.raises(NodeNotFoundError):
        poly_taxonomy.parse_description("Copepoda/Calanidae/Calanus")

    Copepoda_other = poly_taxonomy.parse_description("Copepoda/Other")
    Cyclopoida = poly_taxonomy.parse_description("Copepoda/Cyclopoida", with_alias=True)
    assert Cyclopoida == Copepoda_other

    # Test tag alias
    # TODO: Rework alias handling
    Copepoda_cut = poly_taxonomy.parse_description("Copepoda cut")
    Copepoda_cropped = poly_taxonomy.parse_description(
        "Copepoda cropped", with_alias=True
    )
    assert Copepoda_cropped == Copepoda_cut

    not_root = ~poly_taxonomy.root
    assert isinstance(not_root, NeverDescriptor)

    # Check negation
    Copepoda_not_lateral = poly_taxonomy.parse_description("Copepoda !lateral")
    Copepoda_not_lateral_str = str(Copepoda_not_lateral)
    assert Copepoda_not_lateral_str == "/Copepoda !view:lateral"

    # Check round-tripping of negation
    assert (
        poly_taxonomy.parse_description(Copepoda_not_lateral_str)
        == Copepoda_not_lateral
    )

    # Check implication for negated properties
    Copepoda = poly_taxonomy.root.find_primary("Copepoda")
    male = Copepoda.find_tag(["sex", "male"])
    female = poly_taxonomy.root.find_primary("Copepoda").find_tag(["sex", "female"])
    assert ~male <= female

    assert male.negate(False) == male

    not_male = male.negate()
    assert not_male.negate(False) == not_male

    assert not (Copepoda.negate() <= male)

    Metridia_longa = poly_taxonomy.parse_description("'Metridia longa'")
    not_Calanus_hyperboreus = poly_taxonomy.parse_description(
        "Copepoda !'Calanus hyperboreus'"
    )
    assert not_Calanus_hyperboreus <= Metridia_longa

    # Check update of description
    result = Calanus_male_lateral.copy().add(
        Copepoda_not_lateral, on_conflict="replace"
    )
    expected = poly_taxonomy.parse_description("Calanus sex:male !view:lateral")
    assert result == expected
    assert hash(result) == hash(expected)

    # Test expressions
    Copepoda_not_male_exclude_lateral = poly_taxonomy.parse_expression(
        "Copepoda !sex:male -view:lateral"
    )
    Copepoda_not_male_exclude_lateral.exclude

    query_str = str(Copepoda_not_male_exclude_lateral)
    assert (
        poly_taxonomy.parse_expression(query_str) == Copepoda_not_male_exclude_lateral
    )

    Calanus_female_frontal = poly_taxonomy.parse_description(
        "Calanus sex:female view:frontal"
    )
    assert Copepoda_not_male_exclude_lateral.match(
        Calanus_female_frontal
    ), f"{Copepoda_not_male_exclude_lateral} does not match {Calanus_female_frontal}"

    assert Calanus_male_lateral.copy().add(Copepoda) == Calanus_male_lateral

    for d in Calanus_male_lateral.descriptors:
        assert Copepoda <= d

    assert Calanus_male_lateral.copy().remove(Copepoda) == Description(
        Copepoda.parent or poly_taxonomy.root
    )

    result = Copepoda_not_male_exclude_lateral.apply(Calanus_male_lateral.copy())
    expected = poly_taxonomy.parse_description("Calanus !sex:male")
    assert (
        result == expected
    ), f"{Calanus_male_lateral} %% {Copepoda_not_male_exclude_lateral} != {expected}"

    Cop_wo_Calanus = poly_taxonomy.parse_expression("Copepoda -'Calanus'")
    result = Cop_wo_Calanus.apply(Calanus_male_lateral.copy())
    assert result == poly_taxonomy.parse_description("Copepoda sex:male view:lateral")

    # # TODO: Remove a single descriptor
    # not_lateral = poly_taxonomy.parse_expression("-Copepoda/view:lateral")
    # result = not_lateral.apply(Calanus_male_lateral.copy())
    # assert result == poly_taxonomy.parse_description("Calanus sex:male")

    # Test PolyDescription.remove
    lateral_left = poly_taxonomy.parse_description("Copepoda lateral:left").qualifiers[
        0
    ]
    Calanus_male_lateral_left = poly_taxonomy.parse_description(
        "Calanus male view:lateral:left"
    )
    result = Calanus_male_lateral_left.copy().remove(lateral_left)
    assert result == Calanus_male_lateral

    # Parse expression that excludes negated tags
    poly_taxonomy.parse_expression("Copepoda -!sex:male")


# @pytest.mark.xfail(reason="TODO: Implement hierarchy parsing")
def test_parse_lineage():
    # A concrete example of a polytaxonomy
    poly_taxonomy = PolyTaxonomy.from_dict(taxonomy_dict)

    Calanus_male_lateral = poly_taxonomy.parse_description(
        "Copepoda/Calanus sex:male view:lateral"
    )

    poly_description = poly_taxonomy.parse_lineage(
        ["living", "Crustacea", "Copepoda", "Calanidae", "Calanus", "male+lateral"],
        ignore_unmatched_intermediaries=True,
    )
    assert poly_description == Calanus_male_lateral

    # Assert that missing intermediaries are ignored as long as no suffix is missing
    poly_description = poly_taxonomy.parse_lineage(
        ["living", "Crustacea", "Copepoda", "Calanidae", "male Calanus", "lateral"],
        ignore_unmatched_intermediaries=True,
    )
    assert poly_description == Calanus_male_lateral

    # Check error for unmatched suffix
    with pytest.raises(ValueError):
        poly_taxonomy.parse_lineage(
            [
                "living",
                "Crustacea",
                "Copepoda",
                "Calanidae",
                "male Calanus",
                "lateral",
                "unknown",
            ],
            ignore_unmatched_intermediaries=True,
        )

    # This currently does not work
    Scaphocalanus_cvstage = poly_taxonomy.parse_lineage(
        (
            "cvstage<Scaphocalanus<Scolecitrichidae<Calanoida<Copepoda<Maxillopoda<Crustacea<Arthropoda<Metazoa<Holozoa<Opisthokonta<Eukaryota<living".split(
                "<"
            )[
                ::-1
            ]
        ),
        ignore_unmatched_intermediaries=True,
        with_alias=True,
    )
    assert Scaphocalanus_cvstage == poly_taxonomy.parse_description(
        "Copepoda/Other stage:CV"
    )


def test_description_conflicts():
    # A concrete example of a polytaxonomy
    poly_taxonomy = PolyTaxonomy.from_dict(taxonomy_dict)

    # Update positive tag with negative version
    Copepoda_frontal = poly_taxonomy.parse_description("Copepoda view:frontal")

    with pytest.raises(ConflictError):
        Copepoda_frontal.copy().add(
            poly_taxonomy.parse_description("Copepoda !view:frontal"),
            on_conflict="raise",
        )

    Calanus_hyperboreus = poly_taxonomy.parse_description("'Calanus hyperboreus'")
    d = Calanus_hyperboreus.copy()
    d.add(~d.anchor)
    assert d.anchor == Calanus_hyperboreus.anchor.parent

    result = Copepoda_frontal.copy().add(
        poly_taxonomy.parse_description("Copepoda !view:frontal"),
        on_conflict="replace",
    )
    assert result == poly_taxonomy.parse_description("Copepoda !view:frontal")

    result = Copepoda_frontal.copy().add(
        poly_taxonomy.parse_description("Copepoda !view:frontal"),
        on_conflict="skip",
    )
    assert result == Copepoda_frontal
    assert hash(result) == hash(Copepoda_frontal)

    # Update negative tag with positive version
    Copepoda_not_frontal = poly_taxonomy.parse_description("Copepoda !view:frontal")
    with pytest.raises(ConflictError):
        Copepoda_not_frontal.copy().add(
            poly_taxonomy.parse_description("Copepoda view:frontal"),
            on_conflict="raise",
        )

    result = Copepoda_not_frontal.copy().add(
        poly_taxonomy.parse_description("Copepoda view:frontal"),
        on_conflict="replace",
    )
    assert result == poly_taxonomy.parse_description("Copepoda view:frontal")

    result = Copepoda_not_frontal.copy().add(
        poly_taxonomy.parse_description("Copepoda view:frontal"),
        on_conflict="skip",
    )
    assert result == Copepoda_not_frontal
    assert hash(result) == hash(Copepoda_not_frontal)


def test_poly_taxonomy_binary():
    # A concrete example of a polytaxonomy
    poly_taxonomy = PolyTaxonomy.from_dict(taxonomy_dict)

    poly_taxonomy.fill_indices()
    poly_taxonomy.print_tree()

    cut = poly_taxonomy.parse_description("/ cut").qualifiers[0]
    assert isinstance(cut, TagNode) and cut.name == "cut"
    assert cut.index is not None

    # Get a certain description
    Calanus_male_lateral = poly_taxonomy.parse_description(
        "Copepoda/Calanus sex:male view:lateral"
    )

    assert Calanus_male_lateral.to_binary_str() == {
        "/Copepoda": True,
        "/Copepoda/Calanus": True,
        "/Copepoda/Metridia": False,
        "/Copepoda/Other": False,
        "/Copepoda/view:lateral": True,
        "/Copepoda/view:frontal": False,
        "/Copepoda/view:ventral": False,
        "/Copepoda/sex:male": True,
        "/Copepoda/sex:female": False,
    }

    Copepoda_not_lateral = poly_taxonomy.parse_description("Copepoda !lateral")
    assert Copepoda_not_lateral.to_binary_str() == {
        "/Copepoda": True,
        "/Copepoda/view:lateral": False,
        # TODO: Is this what we want? (`"... view:lateral": False` would be enough)
        "/Copepoda/view:lateral:left": False,
        "/Copepoda/view:lateral:right": False,
    }

    # Check roundtripping of binary description
    Calanus_male_lateral_multilabel = Calanus_male_lateral.to_multilabel(fill_na=0.5)
    assert Calanus_male_lateral == poly_taxonomy.parse_probabilities(
        Calanus_male_lateral_multilabel
    )

    Copepoda_female = poly_taxonomy.parse_description("Copepoda sex:female").qualifiers[
        0
    ]
    assert isinstance(Copepoda_female, TagNode) and Copepoda_female.name == "female"
    assert Copepoda_female.index is not None

    Copepoda_male = poly_taxonomy.parse_description("Copepoda sex:male").qualifiers[0]
    assert isinstance(Copepoda_male, TagNode) and Copepoda_male.name == "male"
    assert Copepoda_male.index is not None

    # Check that max score is used
    Calanus_male_lateral_multilabel = Calanus_male_lateral.to_multilabel(fill_na=0.5)
    # Set score for female high, but male is higher
    Calanus_male_lateral_multilabel[Copepoda_female.index] = 0.95
    assert (
        Calanus_male_lateral_multilabel[Copepoda_female.index]
        < Calanus_male_lateral_multilabel[Copepoda_male.index]
    )
    assert Calanus_male_lateral == poly_taxonomy.parse_probabilities(
        Calanus_male_lateral_multilabel, thr_pos_rel=0.0
    )

    # Check that max score is used
    Calanus_male_lateral_multilabel = Calanus_male_lateral.to_multilabel(fill_na=0.5)
    # Set score for female high, but male is higher
    Calanus_male_lateral_multilabel[Copepoda_female.index] = 0.95
    assert (
        Calanus_male_lateral_multilabel[Copepoda_female.index]
        < Calanus_male_lateral_multilabel[Copepoda_male.index]
    )
    assert Calanus_male_lateral == poly_taxonomy.parse_probabilities(
        Calanus_male_lateral_multilabel, thr_pos_rel=0.0
    )

    # Check that relative threshold prevents descending into the hierarchy
    Calanus_male_lateral_multilabel = Calanus_male_lateral.to_multilabel(fill_na=0.5)
    # Set score for female high, but male is higher
    Calanus_male_lateral_multilabel[Copepoda_female.index] = 0.95
    assert (
        Calanus_male_lateral_multilabel[Copepoda_female.index]
        < Calanus_male_lateral_multilabel[Copepoda_male.index]
    )
    Calanus_lateral = poly_taxonomy.parse_description("Calanus view:lateral")
    assert Calanus_lateral == poly_taxonomy.parse_probabilities(
        Calanus_male_lateral_multilabel, thr_pos_rel=0.5
    )
