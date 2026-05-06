"""Filesystem grounding — detect code references in memory content
and verify they exist on disk.

Used by write_gate to demote candidate memories that mention files
or symbols that don't exist (a common LLM hallucination signal).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Match common code reference patterns:
#   path/to/file.ext   (file with extension, possibly with subdirs)
#   `path/to/file.ext` (backtick-wrapped)
#   foo.bar()           (qualified function call: identifier.identifier())
# Skip bare words like "foo" or "bar".
CODE_REF_PATTERN = re.compile(
    r"""(?:
        (\b\w+\.\w+\(\))                            # qualified function call (priority)
      | `([^`]+\.[a-zA-Z]{1,5})`                    # backtick-wrapped path with ext
      | ([A-Za-z]:[\\/][\w:/\\.\-]+\.[a-zA-Z]{1,5}) # Windows absolute path
      | ([\w/.-]+\.[a-zA-Z]{1,5})\b                 # relative/unix path with ext
    )""",
    re.VERBOSE,
)


# File extensions we recognize as "code" — others (like .png, .jpg) are skipped.
CODE_EXTENSIONS = {
    "py", "ts", "tsx", "js", "jsx", "rs", "go", "java", "kt", "swift",
    "c", "cpp", "h", "hpp", "cs", "rb", "php", "lua", "scala", "clj",
    "sh", "bash", "zsh", "ps1", "psm1",
    "md", "txt", "yml", "yaml", "json", "toml", "ini", "cfg",
    "sql", "html", "css", "scss",
}


def extract_code_refs(content: str) -> list[str]:
    """Extract code references (file paths, function calls) from text.

    Returns a list of unique reference strings, with backticks stripped.
    Filters out non-code extensions (images, archives, etc.).
    """
    if not content:
        return []

    refs: set[str] = set()
    for match in CODE_REF_PATTERN.finditer(content):
        # match.group returns first non-None capture group
        ref = next((g for g in match.groups() if g), None)
        if not ref:
            continue
        # Strip enclosing backticks if any leaked through
        ref = ref.strip("`")
        # Function calls bypass the extension filter
        is_function_call = ref.endswith("()")
        # Filter by extension for file-like refs
        if not is_function_call and "." in ref:
            ext = ref.rsplit(".", 1)[-1].lower()
            if ext.isalpha() and ext not in CODE_EXTENSIONS:
                continue
        refs.add(ref)

    return sorted(refs)


def filesystem_exists(ref: str) -> bool:
    """Check if a code reference resolves to a real file on disk.

    Tries multiple resolution strategies:
      1. Treat ref as absolute path
      2. Resolve against current working directory
      3. Resolve against hermes root (env: HERMES_ROOT or two parents up)

    Function calls (foo.bar()) are NOT checked — only file path refs.
    Returns True if any candidate path exists.
    """
    if not ref:
        return False

    # Strip function-call parens — we don't check those
    if ref.endswith("()"):
        return False

    p = Path(ref)
    if p.is_absolute():
        return p.exists()

    candidates = [
        Path.cwd() / ref,
    ]

    hermes_root = os.environ.get("HERMES_ROOT")
    if hermes_root:
        candidates.append(Path(hermes_root) / ref)
    else:
        # Fallback: assume hermes lives 3 parents up from this file
        # (aisys/memory/grounding.py -> hermes/)
        guess = Path(__file__).resolve().parent.parent.parent
        candidates.append(guess / ref)

    return any(c.exists() for c in candidates)
