from polytaxo.core import ClassNode, NodeNotFoundError
import pytest


def test_find_class():
    Copepoda = ClassNode.from_dict("Copepoda", {"classes": {"Calanoida": {}}})
    Copepoda.find_class("Calanoida")


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


# NB: Virtuals don't have aliases
