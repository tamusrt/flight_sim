"""The Predictions (JARVIS) page and its tools."""

from __future__ import annotations

import json
from typing import Any


def script_json(data: Any) -> str:
    """Return ``data`` as compact JSON that is safe to paste inside a ``<script>`` tag.

    A browser ends a script at the first ``</script`` it sees, even inside a string,
    and ``<!--`` can also confuse it. JSON lets a backslash stand before some
    characters in a string, so ``</`` becomes ``<\\/`` and ``<!--`` becomes
    ``<\\u0021--`` (``\\u0021`` is the code for ``!``). The page reads back
    exactly the same data.
    """
    text = json.dumps(data, separators=(",", ":"))
    return text.replace("</", "<\\/").replace("<!--", "<\\u0021--")
