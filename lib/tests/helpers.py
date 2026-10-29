from polytaxo.core import Description, NegatedRealNode, ClassNode, TagNode
from polytaxo.descriptor import Descriptor
from polytaxo.parser import quote
from polytaxo.taxonomy import Expression


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
