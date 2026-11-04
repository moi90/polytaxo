Taxonomies
==========

Classes and Tags
----------------

A class is a node in a taxonomy that represents a the primary concept of an object.
A tag represents a secondary concept of an object, and is used to annotate objects with additional information.

Homonymy
--------

Homonymy is the phenomenon of two or more nodes in a taxonomy having the same name.
In PolyTaxo, homonymy is allowed, but it is restricted to certain cases where the classes can be unambiguously referenced.

Examples
''''''''

.. admonition:: ✅ Two sibling branches each contain a classes with the same name

    The names are still unambiguous because the two ``X`` nodes can be referenced as ``A/X`` and ``B/X``.

    .. code-block::

        ...
        └── A
            └── X
        └── B
            └── X

.. admonition:: ❌ A top-level class named ``X`` collides with a deeper class also named ``X``

    Here, ``X`` could refer to either ``/X`` or ``/A/X``, which is ambiguous.

    .. code-block::

        ...
        └── X
        └── A
            └── X



.. admonition:: ❌ A class can not have the same name as any of its descendants

    This is ambiguous because ``X`` could refer to either ``/X`` or ``/X/X``.

    .. code-block::

        ...
        └── X
            └── X

