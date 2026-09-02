"""Shared binomial-pattern constants.

Both ``caption_fixer.py`` and ``text_extract.py`` need the same binomial
``Genus species`` regex + English false-positive denylist. Centralizing
them here prevents silent drift between the two modules — the previous
setup had two near-identical hardcoded copies (one per module) plus a
third in a drift-detection test, all of which could be updated in
isolation without the others noticing.
"""
from __future__ import annotations

import re

# Word-boundary binomial pattern: 'Genus species' (lowercase, 3+ chars each).
# Intentionally NO capture groups: ``caption_fixer`` uses ``findall`` (which
# returns tuples when groups exist, breaking its ``str.split()`` logic), and
# ``text_extract`` splits the matched span on whitespace — both flows want
# the full matched substring, not the individual words.
_BINOMIAL_RE = re.compile(r"\b[A-Z][a-z]{3,}\s+[a-z]{3,}\b")

# English false-positive denylist. Both consumers filter these out.
# Lowercase canonical form (string ``.lower()`` is checked against this).
_BINOMIAL_DENY = frozenset({
    'species', 'genera', 'genus', 'sample', 'samples', 'individual',
    'individuals', 'figure', 'figures', 'table', 'caption', 'locality',
    'localities', 'text', 'word', 'words', 'material', 'materials',
    'section', 'plate', 'many', 'most', 'several', 'each',
})
