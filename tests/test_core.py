from polytaxo.core import ClassNode


def test_find_class():
    Copepoda = ClassNode.from_dict("Copepoda", {"classes": {"Calanoida": {}}})
    Copepoda.find_class("Calanoida")
