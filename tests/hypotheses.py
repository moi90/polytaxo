from hypothesis.strategies import (
    DrawFn,
    booleans,
    composite,
    integers,
    lists,
    sampled_from,
)
from polytaxo.core import ClassNode, Description
from polytaxo.taxonomy import Expression
from tests.helpers import format_description, format_expression


@composite
def description(draw: DrawFn, root):
    # Draw one anchor
    class_nodes = [node for node in root.walk() if isinstance(node, ClassNode)]
    anchor: ClassNode = draw(sampled_from(class_nodes))

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
