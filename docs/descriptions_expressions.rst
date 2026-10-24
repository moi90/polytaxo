Descriptions and Expressions
============================

PolyTaxo uses a simple textual syntax to describe and query objects.

The syntax described here is used by :meth:`polytaxo.Taxonomy.parse_description`
and :meth:`polytaxo.Taxonomy.parse_expression`.

Descriptions
------------

Descriptions are the core of PolyTaxo's semantics. They are used to annotate objects with their properties and characteristics.

.. admonition:: Example

    .. math::
        \underbrace{\texttt{Copepoda/Calanus}}_{\text{Anchor class}}\ \underbrace{\texttt{view:lateral !truncated stage:CVI:female:ovigerous}}_{\text{Qualifiers}}

    The thusly annotated object is a female ovigerous Calanus copepod (of unspecified species) in the CVI stage, viewed laterally. The image is not truncated, i.e. the entire animal is visible.

- A description string consists of a series of *descriptors* (separated by whitespace).
- The first descriptor is the *anchor class* which specifies the main concept of the object.
- The anchor class is followed by zero or more *qualifiers* which further specify the object.
- Qualifiers are resolved relative to the anchor class.
- Descriptor names can be *disambiguated* by specifying their parents in the taxonomy hierarchy. 
- Qualifiers can be *negated* by prefixing them with ``!``.

Negation
~~~~~~~~

The distinction between affirmed, negated, and unspecified descriptors is central to PolyTaxo's semantics. It allows for precise descriptions of objects, including what is known and what is explicitly excluded.

- ``a`` means the concept ``a`` is explicitly present / true.
- ``!a`` means the concept ``a`` is explicitly absent / false.
- If neither ``a`` nor ``!a`` is present, the concept is unspecified.
- Negation applies to the next descriptor only.
    
.. admonition:: More Examples

    .. list-table::
        :header-rows: 1

        * - Description
          - Meaning
        * - ``Copepoda sex:male``
          - male is affirmed
        * - ``Copepoda !sex:male``
          - male is negated
        * - ``Copepoda``
          - sex is unspecified

Disambiguated Name
~~~~~~~~~~~~~~~~~~

The name of a descriptor can be short (just the class or tag name) or include parent names when needed to distinguish tags or classes with the same name.

For classes, disambiguation is necessary in the case of taxonomic homonymy, where the same name is used in different branches of the taxonomy. One example is the genus name ``Turbinaria``, which is used for both a brown algae and a coral genus.

For tags, disambiguation allows for the same tag name to be used in different contexts, e.g., a descendant might redefine a tag with the same name as one of its ancestors.

- ``/`` separates class segments.
- A leading ``/`` denotes an absolute path from the taxonomy root.
- ``:`` separates tag segments.

.. admonition:: Examples

    .. list-table::
        :header-rows: 1

        *   - Form
            - Example
            - Meaning
        *   - Short class name
            - ``Turbinaria``
            - Ambiguous in the case of homonymy.
        *   - Disambiguated class name
            - ``Sargassaceae/Turbinaria``
            - Identifies the brown algae genus Turbinaria unambiguously.
        *   - Disambiguated class name
            - ``Cnidaria/Turbinaria``
            - Identifies the coral genus Turbinaria unambiguously.
        *   - Short tag name
            - ``high``
            - In an example where both ``turbidity:high`` and ``contrast:high`` exist, this short form is ambiguous.
        *   - Disambiguated tag name
            - ``turbidity:high``
            - High turbidity.
        *   - Disambiguated tag name
            - ``contrast:high``
            - High contrast.

Quoted Names
~~~~~~~~~~~~

Quoted names allow for the use of names that contain special characters or whitespace.
Quoting applies to a full disambiguated name.

.. admonition:: Example

    .. list-table::
        :header-rows: 1

        *   - Example
            - Meaning

        *   - ``'Turbinaria ornata'``
            - Quoted name containing whitespace.
        *   - ``'Copepoda/Calanus hyperboreus'``
            - Quoted disambiguated class name where one segment contains whitespace.

Expressions
-----------

Expressions are used to query objects based on their descriptions.

- Expression strings are a superset of description strings, i.e., any valid description is also a valid expression.
- Expressions additionally support *exclusion* clauses prefixed by ``-``.

Exclusion
~~~~~~~~~

Exclusion clauses allow for the exclusion of specific descriptors or descriptions from the query results.

- ``-`` starts an exclusion clause.
- ``-`` before a descriptor excludes a single descriptor.
- ``-(...)`` excludes a full parenthesized description.
- ``-a`` means matches with affirmed ``a`` are excluded.
- ``-!a`` means matches with negated ``a`` are excluded.

.. admonition:: Examples

    .. list-table::
        :header-rows: 1

        *   - Expression
            - Meaning
        *   - ``Copepoda !sex:male -view:lateral``
            - Match copepods that are explicitly not male, excluding laterally viewed ones.
        *   - ``Copepoda -Calanus``
            - Match copepods, excluding those annotated as Calanus.
        *   - ``Copepoda -(sex:male view:lateral)``
            - Match copepods, excluding those that are both male and lateral.
        *   - ``Copepoda -!sex:male``
            - Match copepods, excluding those that are explicitly not male.



Formal Grammar
--------------

..
    NB: The following grammar should be kept in sync with the description above and the implementation in :meth:`polytaxo.Taxonomy.parse_description` and :meth:`polytaxo.Taxonomy.parse_expression`.

.. productionlist::
    ws: (space | tab | newline)+
    unquoted_class_name: "/"? name ( "/" name)*
    quoted_class_name : ( "'" unquoted_class_name "'" ) | ( '"' unquoted_class_name '"' )
    class_name: unquoted_class_name | quoted_class_name
    unquoted_tag_name: name ( ":" name)*
    quoted_tag_name : ( "'" unquoted_tag_name "'" ) | ( '"' unquoted_tag_name '"' )
    tag_name: unquoted_tag_name | quoted_tag_name
    class_or_tag_name: class_name | tag_name
    negation: "!"
    descriptor: negation* class_or_tag_name
    description: class_name ( ws descriptor )*
    excluded_description: "(" descriptor ( ws descriptor )* ")"
    exclude_clause: "-" (descriptor | excluded_description)
    expression: class_name ( ws (descriptor | exclude_clause ) )*
