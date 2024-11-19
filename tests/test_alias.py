from polytaxo.alias import calc_specificy, Alias


def test_calc_specificy():
    assert calc_specificy("*") == 1
    assert calc_specificy("Foo?") == 4
    assert calc_specificy("[Foo]?") == 1
    assert calc_specificy("[!Foo]?") == 1
    assert calc_specificy("[!Foo]? Bar") == 5


def test_Alias():
    assert Alias("*").match("Foo") == 1
    assert Alias("*").match("") == 1
    assert Alias("Foo").match("Foo") == 4
    assert Alias("Foo").match("Bar") == 0
    assert Alias("*Foo").match("Foo") == 4
    assert Alias("*Foo").match("BarFoo") == 4
    assert Alias("BarFoo").match("BarFoo") == 7
