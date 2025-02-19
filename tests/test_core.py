from polytaxo.core import ClassNode


def test_find_primary():
    Copepoda = ClassNode.from_dict("Copepoda", {"classes": {"Calanoida": {}}})
    Copepoda.find_primary("Calanoida")
