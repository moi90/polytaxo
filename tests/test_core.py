from polytaxo.core import PrimaryNode


def test_find_primary():
    Copepoda = PrimaryNode.from_dict("Copepoda", {"children": {"Calanoida": {}}})
    Copepoda.find_primary("Calanoida")
