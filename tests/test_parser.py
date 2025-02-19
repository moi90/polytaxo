from hypothesis import given
from hypothesis.strategies import (
    DrawFn,
    booleans,
    composite,
    integers,
    lists,
    sampled_from,
)

from polytaxo.core import Description, NegatedRealNode, ClassNode, TagNode
from polytaxo.descriptor import Descriptor
from polytaxo.parser import quote, tokenize
from polytaxo.taxonomy import Expression, Taxonomy
from tests.data import taxonomy_dict

taxonomy = Taxonomy.from_dict(taxonomy_dict)


def format_descriptor_quoted(d: Descriptor, anchor=None, quote_chars="'\"") -> str:
    if isinstance(d, (ClassNode, TagNode)):
        return quote(d.format(anchor=anchor), quote_chars=quote_chars)
    if isinstance(d, NegatedRealNode):
        return "!" + format_descriptor_quoted(
            d.node, anchor=anchor, quote_chars=quote_chars
        )
    raise ValueError(f"Unexpected {d!r}")


def format_description(d: Description, anchor=None, space=" ", quote_chars="'\""):
    """
    A configurable version of `Description.__str__` for testing purposes.
    """
    return space.join(
        [format_descriptor_quoted(d.anchor, anchor=anchor, quote_chars=quote_chars)]
        + sorted(
            [
                format_descriptor_quoted(q, anchor=d.anchor, quote_chars=quote_chars)
                for q in d.qualifiers
            ]
        )
    )


def format_expression(e: Expression, space=" ", quote_chars="'\""):
    """
    A configurable version of `Expression.__str__` for testing purposes.
    """

    if not e.exclude:
        return format_description(e.include, space=space, quote_chars=quote_chars)

    exclude = []
    for excl in e.exclude:
        if isinstance(excl, Descriptor):
            exclude.append(
                f"-{format_descriptor_quoted(excl, e.include.anchor, quote_chars=quote_chars)}"
            )
        else:
            exclude.append(
                f"-({format_description(excl, e.include.anchor, space=space, quote_chars=quote_chars)})"
            )

    return (
        format_description(e.include, space=space, quote_chars=quote_chars)
        + space
        + space.join(exclude)
    )


@composite
def description(draw: DrawFn, root):
    # Draw one anchor
    primary_nodes = [node for node in root.walk() if isinstance(node, ClassNode)]
    anchor: ClassNode = draw(sampled_from(primary_nodes))

    # Draw zero or more qualifiers applicable to this anchor
    node = anchor
    tags = []
    while node is not None:
        for t in node.tags:
            tags.extend(t.walk())
        node = node.parent
    tags.extend([t.negate() for t in tags])
    qualifiers = draw(lists(sampled_from(tags)))

    description = Description(anchor)
    description.update(qualifiers, "replace")
    return description


@composite
def description_formatted(draw: DrawFn, root):
    descr = draw(description(root))
    if draw(booleans()):
        space = draw(sampled_from([" ", "  "]))
        quote_chars = draw(sampled_from(["'\"", "\"'"]))
        formatted = format_description(descr, space=space, quote_chars=quote_chars)
    else:
        formatted = str(descr)

    return (descr, formatted)


@given(description_formatted(taxonomy.root))
def test_parse_description(description_formatted):
    description, description_str = description_formatted
    description2 = Description.from_string(taxonomy.root, description_str)
    assert description == description2, f"{description} vs. {description2}"


@given(description_formatted(taxonomy.root))
def test_parse_expression_simple(description_formatted):
    description, description_str = description_formatted
    expression = Expression.from_string(taxonomy.root, description_str)
    assert description == expression.include, f"{description} vs. {expression.include}"


@composite
def expression(draw: DrawFn, root):
    include = draw(description(root))
    exclude = [
        draw(description(include.anchor).filter(lambda d: d.anchor != include.anchor))
        for _ in range(draw(integers(min_value=0, max_value=3)))
    ]
    return Expression(include, exclude)


@composite
def expression_formatted(draw: DrawFn, root):
    expr = draw(expression(root))

    if draw(booleans()):
        space = draw(sampled_from([" ", "  "]))
        quote_chars = draw(sampled_from(["'\"", "\"'"]))
        formatted = format_expression(expr, space=space, quote_chars=quote_chars)
    else:
        formatted = str(expr)
    return (expr, formatted)


@given(expression_formatted(taxonomy.root))
def test_parse_expression_complex(expression_formatted):
    expression, expression_str = expression_formatted
    expression2 = Expression.from_string(taxonomy.root, expression_str)
    assert (
        expression == expression2
    ), f"<{expression}> vs. <{expression2}>, {tokenize(expression_str)}"
