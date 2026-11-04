from polytaxo.core import ClassNode, NodeNotFoundError, TagNode
import pytest


def test_find_class():
    Copepoda = ClassNode.from_dict("Copepoda", {"classes": {"Calanoida": {}}})
    Copepoda.find_class("Calanoida")


def test_walk_include_self_toggle():
    """walk() should include `self` by default and skip it when include_self=False."""
    root = ClassNode.from_dict(
        "Root",
        {
            "classes": {"ClassA": {}},
            "tags": {"TagA": {"tags": {"TagB": {}}}},
        },
    )

    with_self = [node.path for node in root.walk()]
    without_self = [node.path for node in root.walk(include_self=False)]

    assert with_self == [
        ("Root",),
        ("Root", "ClassA"),
        ("Root", "TagA"),
        ("Root", "TagA", "TagB"),
    ]
    assert without_self == with_self[1:]


def test_find_all_class_respects_anchored_alias_active_path():
    root = ClassNode.from_dict(
        "Root",
        {
            "classes": {
                "Parent": {
                    "classes": {
                        "AnchoredChild": {"alias": "/anchored"},
                    }
                },
            }
        },
    )

    with pytest.raises(NodeNotFoundError):
        root.find_class(("anchored",), with_alias=True)

    assert (
        root.find_class(("Parent", "anchored"), with_alias=True).name == "AnchoredChild"
    )


def test_find_all_tag_respects_anchored_alias_active_path():
    root = ClassNode.from_dict(
        "Root",
        {
            "tags": {
                "view": {
                    "tags": {
                        "lateral": {
                            "tags": {
                                "left": {"alias": "/lefty"},
                            }
                        }
                    }
                }
            }
        },
    )

    view = root.tags[0]

    with pytest.raises(NodeNotFoundError):
        view.find_tag(("lefty",), with_alias=True)

    assert view.find_tag(("lateral", "lefty"), with_alias=True).name == "left"


def test_direct_child_anchored_alias_matches_from_active_anchor():
    # Regression: when lookup starts at an already-active anchor, anchored aliases on
    # its direct children must still match for single-name lookups.
    root = ClassNode.from_dict(
        "Root",
        {
            "classes": {
                "Calanus": {
                    "classes": {
                        "Calanus other": {
                            "alias": "/*",
                        }
                    }
                }
            }
        },
    )

    calanus = root.find_class("Calanus")
    calanus_glacialis = calanus.find_class("Calanus glacialis", with_alias=True)
    assert calanus_glacialis.name == "Calanus other"


def test_direct_child_anchored_tag_alias_matches_from_active_anchor():
    # Regression: when lookup starts at an already-active tag anchor, anchored aliases
    # on direct child tags must still match for single-name lookups.
    root = ClassNode.from_dict(
        "Root",
        {
            "tags": {
                "view": {
                    "tags": {
                        "other": {
                            "alias": "/*",
                        }
                    }
                }
            }
        },
    )

    view = root.find_tag("view")
    lefty = view.find_tag("lefty", with_alias=True)
    assert lefty.name == "other"


def test_add_tag_rejects_case_insensitive_duplicate_sibling_name():
    root = ClassNode.from_dict("Root", {"tags": {"group": {}}})
    group = root.find_tag("group")

    group.add_tag(TagNode("X", group, None))

    with pytest.raises(ValueError, match="same name already exists"):
        group.add_tag(TagNode("x", group, None))


# NB: Virtuals don't have aliases


def test_add_class_does_not_shadow_existing_class():
    """Test that adding a class with the same name as an existing class in an ancestor raises an error.

    This is allowed (because ):
    ...
    └── A
        └── X
    └── B
        └── X

    This is not allowed (because it would be unclear if "X" refers to .../X or .../A/X):
    ...
    └── X
    └── A
        └── X
    """
    root = ClassNode("", None, None)
    A = root.add_class(ClassNode("A", root, None))
    B = A.add_class(ClassNode("B", A, None))

    # A/X, then A/B/X
    A.add_class(ClassNode("X", A, None))

    with pytest.raises(ValueError):
        B.add_class(ClassNode("X", B, None))

    # A/B/X, then A/X
    B.add_class(ClassNode("Y", B, None))

    with pytest.raises(ValueError):
        A.add_class(ClassNode("Y", A, None))
