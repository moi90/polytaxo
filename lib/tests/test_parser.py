from typing import List, Tuple

import hypothesis.strategies as st
from hypothesis import given, note

from polytaxo.core import Description
from polytaxo.parser import tokenize
from polytaxo.taxonomy import Expression, Taxonomy
from tests.data import taxonomy_dict
from tests.hypotheses import any_taxonomy, description_formatted, expression_formatted

taxonomy = Taxonomy.from_dict(taxonomy_dict)


@given(st.data())
def test_parse_description(data):
    _taxonomy: Taxonomy = data.draw(
        st.one_of(st.just(taxonomy), any_taxonomy()),
    )

    note(f"Taxonomy:\n{_taxonomy}")

    description, description_str = data.draw(description_formatted(_taxonomy.root))

    description2 = Description.from_string(_taxonomy.root, description_str)
    assert description == description2, f"{description} vs. {description2}"


@given(description_formatted(taxonomy.root))
def test_parse_expression_simple(description_formatted):
    description, description_str = description_formatted
    expression = Expression.from_string(taxonomy.root, description_str)
    assert description == expression.include, f"{description} vs. {expression.include}"


@given(expression_formatted(taxonomy.root))
def test_parse_expression_complex(expression_formatted):
    expression, expression_str = expression_formatted
    expression2 = Expression.from_string(taxonomy.root, expression_str)
    assert expression == expression2, (
        f"<{expression}> vs. <{expression2}>, {tokenize(expression_str)}"
    )


@given(st.lists(description_formatted(taxonomy.root), min_size=4))
def test_tokenize_parallel(descriptions_formatted: List[Tuple[Description, str]]):
    """
    Test that the tokenizer does work in parallel.
    """

    descriptions, description_strs = zip(*descriptions_formatted)

    from multiprocessing.pool import ThreadPool

    with ThreadPool(len(descriptions)) as pool:
        descriptions2 = tuple(
            pool.map(
                lambda description_str: Description.from_string(
                    taxonomy.root, description_str
                ),
                description_strs,
            )
        )

    assert descriptions == descriptions2
