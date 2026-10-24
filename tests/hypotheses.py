from hypothesis.strategies import (
    DrawFn,
    booleans,
    composite,
    integers,
    lists,
    sampled_from,
    from_regex,
)
from polytaxo.core import ClassNode, Description, NodeNotFoundError, TagNode
from polytaxo.taxonomy import Expression, Taxonomy
from tests.helpers import format_description, format_expression


@composite
def any_taxonomy(draw: DrawFn) -> Taxonomy:
    """
    Construct a random taxonomy with a few levels of children, each with some tags.
    """

    # Create a root node
    root = ClassNode("", None, None)

    def add_tag_children(node: ClassNode | TagNode, depth):
        if depth <= 0:
            return

        # Draw a set of unique tag names for this node
        names = draw(
            lists(
                from_regex(r"[a-zA-Z_][-_ a-zA-Z0-9]*", fullmatch=True),
                min_size=0,
                max_size=3,
                unique_by=str.casefold,
            )
        )
        for name in names:
            child = node.add_tag(TagNode(name, node, None))
            add_tag_children(child, depth - 1)

    def tag_not_found(node: ClassNode, name: str):
        try:
            node.find_tag(name)
        except NodeNotFoundError:
            return True
        return False

    def add_class_children(node: ClassNode, depth):
        if depth <= 0:
            return

        # Draw a set of unique class names for this node
        names = draw(
            lists(
                from_regex(r"[a-zA-Z_][-_ a-zA-Z0-9]*", fullmatch=True)
                # Filter out any names that are already available as tags for this node
                .filter(lambda name: tag_not_found(node, name)),
                min_size=0,
                max_size=3,
                unique_by=str.casefold,
            )
        )
        for name in names:
            child = node.add_class(ClassNode(name, node, None))
            add_tag_children(child, depth - 1)
            add_class_children(child, depth - 1)

    add_class_children(root, 3)

    return Taxonomy(root)


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

    if tags:
        tags.extend([t.negate() for t in tags])
        qualifiers = draw(lists(sampled_from(tags)))
    else:
        qualifiers = []

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
