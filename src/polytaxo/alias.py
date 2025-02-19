import fnmatch
import re

# *       matches everything
# ?       matches any single character
# [seq]   matches any character in seq
# [!seq]  matches any char not in seq
_SHELL_PATTERNS = re.compile(r"\*|\?|(\[[^\]]*\])")


def calc_specificy(pattern: str):
    """Calculate the specificy of a shell style pattern. Always >0."""
    # Replace all shell pattern parts by ""
    pattern = _SHELL_PATTERNS.sub("", pattern)
    return 1 + len(pattern)


class Alias:
    def __init__(self, pattern: str) -> None:
        self.pattern = pattern.casefold()
        self._match = re.compile(fnmatch.translate(self.pattern)).match
        self.specificy = calc_specificy(self.pattern)

    def match(self, name: str) -> int:
        """Test whether `name` matches `pattern` and return specificy."""
        name = name.casefold()
        if self._match(name):
            return self.specificy
        return 0

    def __repr__(self) -> str:
        return f"Alias({self.pattern!r})"
