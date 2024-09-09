import re
import shlex


def _tokenize_expression_str(query_str: str):
    """Tokenize a string expression."""
    # Split into parts, then at : and / and separate off !
    return [
        tuple(filter(None, (re.split("/|:|(!)|(-)", part))))
        for part in shlex.split(query_str)
    ]
