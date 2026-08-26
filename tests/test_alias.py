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


def test_anchored_alias_matches_only_on_active_path():
    alias = Alias("/Calanus")

    assert alias.match("Calanus", has_active_parent=False) == 0
    assert alias.match("Calanus", has_active_parent=True) > 0


def test_anchored_alias_keeps_pattern_semantics_when_active():
    alias = Alias("/*Foo")

    assert alias.match("BarFoo", has_active_parent=False) == 0
    assert alias.match("BarFoo", has_active_parent=True) > 0
