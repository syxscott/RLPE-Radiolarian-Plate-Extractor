from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, replace
from typing import Any

logger = logging.getLogger(__name__)

AGE_PATTERN = re.compile(
    r"\b(?:Early|Middle|Late|Lower|Upper)\s+[A-Z][a-z]+|"
    r"\b(?:Precambrian|Cambrian|Ordovician|Silurian|Devonian|Carboniferous|Permian|"
    r"Triassic|Jurassic|Cretaceous|Paleocene|Eocene|Oligocene|Miocene|Pliocene|"
    r"Pleistocene|Holocene)\b",
    re.IGNORECASE,
)
FORMATION_PATTERN = re.compile(r"\b([A-Z][A-Za-z\-\s]+(?:Formation|Member|Group|Fm\.|Mb\.|Gp\.))\b")
# Locality phrase: "from/at/in <Capitalised name>" — bounded so the
# match doesn't run past the immediate locality token. The previous
# pattern ``[A-Za-z\-\s]{2,80}`` was too permissive: ``\s`` allowed
# arbitrary cross-sentence runs (e.g. "from California is the type"
# matched "California is the type"). Now: capture a Capitalised word
# optionally followed by 1-3 more Capitalised tokens (proper-noun
# locality phrase like "Southwest Japan", "New South Wales"); bail
# out at the first lowercase word, end-of-sentence punctuation, or
# numeric token.
LOCALITY_PATTERN = re.compile(
    r"\b(?:from|at|in)\s+"
    r"([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,3})"
    r"(?=\s*[,.;:()]|\s+(?:and|the|of|a|an|is|are|was|were|in|at)\b|$)"
)
COORDINATE_PATTERN = re.compile(
    r"\b(\d{1,3}(?:\.\d+)?)\s*°?\s*([NSns])?[,\s]+(\d{1,3}(?:\.\d+)?)\s*°?\s*([EWew])?\b"
)

# Round 18 audit: split the FORMATION_PATTERN output into the three
# stratigraphic ranks it actually covers (Group / Formation / Member)
# and add lithology / biozone / country extraction so the published
# GeologyLinkRecord fills more of its 25 declared fields.

# Match the three ranks independently so a single string like
# "Sicanian Group → Rosso Ammonitico Formation → Lower Member" parses
# into {group, formation, member} in one pass.
#
# Round 18 audit: the previous regex used ``[A-Z][A-Za-z\-]+``
# (greedy, 0-3 more words) which matched across sentence boundaries
# and produced garbage like "RAM is the Fonzaso Formation" — the
# regex would start matching at "Rosso" or "Medio" and walk forward
# to "Formation" while eating every word in between. The fix uses a
# much narrower ``[A-Z][A-Za-z\-]{0,30}`` with a non-greedy ``?``
# quantifier and explicit word boundaries so the match terminates at
# the first lowercase word or sentence end.
_GROUP_RE = re.compile(
    r"\b([A-Z][A-Za-z\-]{0,30}?\s+(?:Group|Gp\.))\b"
)
_FORMATION_RE = re.compile(
    r"\b([A-Z][A-Za-z\-]{0,30}?\s+(?:Formation|Fm\.))\b"
)
_MEMBER_RE = re.compile(
    r"\b([A-Z][A-Za-z\-]{0,30}?\s+(?:Member|Mb\.))\b"
)

# Lithology dictionary: lowercase match against a curated set of
# sedimentary / volcanic / biogenic rock names that appear in
# radiolarian-paper captions. ``\b`` boundaries prevent
# "siliceous" matching inside "siliceously".
_LITHOLOGY_TERMS = (
    "siliceous limestone", "limestone", "chert", "radiolarian chert",
    "bedded chert", "ribbon chert", "cherty limestone",
    "marl", "marlstone", "shale", "mudstone", "claystone",
    "sandstone", "siltstone", "conglomerate", "breccia",
    "dolomite", "dolostone", "micrite", "biomicrite",
    "calcarenite", "calcilutite", "tuff", "tuffaceous",
    "basalt", "basaltic", "andesite", "rhyolite",
    "ophiolite", "serpentinite", "chalk", "marlstone",
    "glauconitic sandstone", "phosphorite", "ironstone",
    "black shale", "organic-rich shale",
)
LITHOLOGY_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in _LITHOLOGY_TERMS) + r")\b",
    re.IGNORECASE,
)

# Round 24 (user audit): paleoenvironment / redox / chemostrat /
# facies dictionaries. Each is a curated vocabulary of 20-30
# terms chosen to be cross-referenced against standard
# sedimentology literature. The regex matches case-insensitive
# whole-word. The first match wins (we keep the operator focused
# on the dominant signal, not every mention).
#
# ``paleoenvironment`` answers "how oxygenated was the water
# column?"  e.g. ``anoxic``, ``euxinic``, ``oxic``, ``suboxic``,
# ``dysoxic``, ``upwelling zone``, ``restricted basin``.
# Critical for P/T boundary research: anoxia / euxinia is one of
# the leading kill mechanisms for radiolarians.
_PALEOENV_VOCAB = (
    "anoxic", "euxinic", "oxic", "suboxic", "dysoxic", "suboxic-anoxic",
    "oxygen minimum zone", "OMZ", "upwelling", "upwelling zone",
    "restricted basin", "open marine", "pelagic", "hemipelagic",
    "neritic", "littoral", "shallow marine", "deep marine",
    "near-shore", "slope", "basinal", "abyssal", "photic zone",
    "aphotic zone", "dysoxic bottom water",
)
PALEOENV_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in _PALEOENV_VOCAB) + r")\b",
    re.IGNORECASE,
)

# ``redox`` uses the Algeo & Tribovillard (2009) classification:
# oxic / dysoxic / suboxic / anoxic / euxinic.
_REDOX_VOCAB = (
    "oxic", "dysoxic", "suboxic", "anoxic", "euxinic",
    "ferruginous", "sulfidic", "anoxic-ferruginous", "anoxic-sulfidic",
    "non-sulfidic anoxic",
)
REDOX_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in _REDOX_VOCAB) + r")\b",
    re.IGNORECASE,
)

# ``chemostrat`` covers carbon-isotope excursions, mass
# extinctions, and other named chemostratigraphic events.
# Critical for P/T boundary: a δ¹³C negative excursion marks
# the extinction horizon.
_CHEMOSTRAT_VOCAB = (
    "CIE", "carbon isotope excursion", "δ13C excursion",
    "delta 13C excursion", "delta-13C excursion", "C-isotope excursion",
    "mass extinction", "biocalcification crisis", "LIP", "large igneous province",
    "Siberian Traps", "TE disaster", "oceanic anoxic event", "OAE",
    "bonarelli event", "toarcian OAE", "cenomanian-turonian OAE",
    "Frasnian-Famennian boundary", "Hangenberg event", "P/T boundary",
    "end-Triassic extinction", "end-Permian extinction",
    "end-Guadalupian extinction",
    "Permian-Triassic boundary", "carbon isotope negative excursion",
    "δ13C negative excursion", "strontium isotope excursion",
    "osmium isotope excursion", "mercury anomaly",
)

# Round 25 (user audit follow-up): the user asked specifically
# for "geochemistry & paleoenvironment proxies" and highlighted
# that for P/T boundary research the core data points are
# the δ¹³C values themselves (not just the EVENT name). Round 25
# adds a regex that captures the numeric value of a stable
# isotope measurement: ``δ13C = -3.2 ‰`` or ``δ18O = +5.1 ‰`` or
# ``87Sr/86Sr = 0.70712``. These are stored as additional
# chemostrat proxies so the operator can run numeric comparisons
# without re-parsing the paper.
#
# Patterns:
#   - δ13C = -3.2 ‰ / δ13C: -3.2 / δ13C -3.2 permil
#   - δ18O = +5.1 ‰
#   - 87Sr/86Sr = 0.70712 (isotope ratio)
#   - TOC = 4.5 wt% (total organic carbon)
#   - 13C isotope excursion of -2 ‰
#
# Each pattern is captured into the geology link's
# ``evidence_text`` so the operator can see the exact sentence
# from which the value was extracted. We don't try to store
# the value itself as a separate field because that would
# require schema changes (multiple values per record); the
# operator can grep the evidence_text instead.
_ISOTOPE_PATTERN = re.compile(
    r"(?:"
    # δ13C / δ18O / δ34S — with optional sign, value, permil
    r"\b[δd](?:13C|18O|34S|15N|8[78]Sr)\s*"
    r"(?:=|:)?\s*[-+]?\d+(?:\.\d+)?\s*[‰%]*"
    r"|"  # 87Sr/86Sr (or 86/86, 87/87, 88/86) ratio
    r"\b8[67]Sr/8[67]Sr\s*(?:=|:)?\s*0\.\d{3,6}"
    r"|"  # TOC = 4.5 wt%
    r"\bTOC\s*(?:=|:)?\s*\d+(?:\.\d+)?\s*(?:wt%|%)"
    r"|"  # Hg anomaly (mercury, ppb)
    r"\bHg(?:\s+anomaly)?\s*(?:=|:)?\s*\d+(?:\.\d+)?\s*ppb"
    r")",
    re.IGNORECASE,
)
CHEMOSTRAT_PATTERN = re.compile(
    r"(?:" + "|".join(re.escape(t) for t in _CHEMOSTRAT_VOCAB) + r")",
    re.IGNORECASE,
)

# ``facies`` is the standard sedimentological facies vocabulary.
# We support both descriptive terms and named lithofacies.
_FACIES_VOCAB = (
    "turbidite", "turbiditic", "calciturbidite", "debrites",
    "pelagic", "hemipelagic", "neritic", "littoral", "shallow water",
    "deep water", "deep-sea", "abyssal", "slope", "shelf", "platform",
    "carbonate platform", "rimmed platform", "isolated platform",
    "basinal", "basin", "back-arc basin", "fore-arc basin",
    "intra-arc basin", "rift basin", "passive margin",
    "active margin", "subduction zone", "accretionary wedge",
    "foreshore", "shoreface", "offshore", "deep-water",
    "basin plain", "distal turbidite", "proximal turbidite",
)
FACIES_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in _FACIES_VOCAB) + r")\b",
    re.IGNORECASE,
)

# Biozone patterns:
#   - "N. optima Zone" / "P. uvus Zone" (radiolarian first-letter abbrev.)
#   - "Zone 5" / "Subzone 5a" / "Zone 5a" (numbered biozones — the
#     existing pattern required an uppercase-letter-then-optional-
#     lowercase suffix, so "Zone 5" without a trailing letter did
#     NOT match. Fixed below.)
#   - "Morozovella aragonensis Zone" (full taxon-named zone)
#   - "UAZ 1" / "UAZ 1-7" / "UAZ 12" — Unitary Association Zones
#     (Baumgartner et al. 1995 standard for Mesozoic radiolarians).
#   - "Pessagno Zone A" / "Pessagno Zone 1" — Pessagno's
#     Jurassic-Cretaceous zonation (Pessagno 1977).
#   - "Pessagno Zones A-B" — range form of Pessagno zones.
_BIOZONE_RE = re.compile(
    r"\b(?:"
    # First-letter abbrev.: "N. optima Zone", "P. uvus Zone"
    r"[A-Z]\.\s*[a-z]+\s+Zone"
    r"|"
    # Numbered zone: "Zone 5", "Subzone 5a", "Zone 5a". The trailing
    # [a-z]? is OPTIONAL — "Zone 5" must match.
    r"(?:Sub)?[Zz]one\s+\d+[a-z]?"
    r"|"
    # Full taxon-named zone: "Morozovella aragonensis Zone"
    r"[A-Z][a-z]+\s+[a-z]+\s+Zone"
    r"|"
    # UAZ-numbered zones: "UAZ 1", "UAZ 12", "UAZ 1-7".
    r"UAZ\s+\d+(?:-\d+)?"
    r"|"
    # Pessagno zones: "Pessagno Zone A", "Pessagno Zone 1",
    # "Pessagno Zones A-B".
    r"Pessagno\s+[Zz]ones?\s+[A-Z\d](?:-\d|[a-z])?"
    r")\b"
)

# Country list: ISO 3166-1 shortlist of countries that appear in
# radiolarian-paper locality phrases. The list is intentionally
# short — matches against a curated set beat a fuzzy geo-lookup for
# "Italy" vs "Italian" (the regex boundary handles the difference).
# ``Sicily`` etc. are deliberately excluded: they're regions, not
# countries, and including them caused "Sicily" to wrongly match
# before "Italy" in the Beccaro paper.
_COUNTRIES = (
    "Italy", "Japan", "China", "Turkey", "Greece", "Oman",
    "New Zealand", "Australia", "Austria", "France", "Germany",
    "Spain", "Portugal", "Swiss", "Switzerland", "Russia",
    "Canada", "USA", "United States", "Mexico", "Argentina",
    "Chile", "Brazil", "India", "Pakistan", "Philippines",
    "Indonesia", "Iran", "Iraq", "Saudi Arabia", "South Africa",
    "Egypt", "Tunisia", "Morocco", "Algeria", "Norway", "Sweden",
    "Finland", "Denmark", "Poland", "Czech Republic", "Hungary",
    "Romania", "Bulgaria", "Greece", "Turkey", "Cyprus",
)

# Round 21: country centroid fallback table. When a paper mentions
# only a country name (e.g. "Greece", "Tunisia") without explicit
# coordinates, the older pipeline returned ``modern_latitude=None``
# because ``_extract_first_coord`` requires hemisphere or degree
# symbol. We now look up the country centroid as a low-confidence
# fallback. Centroid coordinates are approximate (a single point
# representing the country), so they're tagged with
# ``confidence=0.3`` and ``coordinate_source="country_centroid"`` so
# the operator can tell centroid-derived coords from regex-derived
# ones.
#
# This is intentionally a SHORT list — only countries that appear in
# radiolarian papers. For countries NOT in this table the older
# behaviour (None coords) applies. A future round could swap this
# for a GeoNames / OpenStreetMap gazetteer lookup.
_COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "Italy": (41.5, 12.5),
    "Japan": (36.0, 138.0),
    "China": (35.0, 105.0),
    "Turkey": (39.0, 35.0),
    "Greece": (39.0, 22.0),
    "Oman": (21.0, 57.0),
    "New Zealand": (-41.0, 174.0),
    "Australia": (-25.0, 133.0),
    "Austria": (47.5, 14.5),
    "France": (46.0, 2.0),
    "Germany": (51.0, 9.0),
    "Spain": (40.0, -4.0),
    "Portugal": (39.5, -8.0),
    "Switzerland": (46.8, 8.2),
    "Russia": (60.0, 100.0),
    "Canada": (60.0, -95.0),
    "USA": (40.0, -100.0),
    "United States": (40.0, -100.0),
    "Mexico": (23.0, -102.0),
    "Argentina": (-34.0, -64.0),
    "Chile": (-35.0, -71.0),
    "Brazil": (-14.0, -51.0),
    "India": (20.0, 77.0),
    "Pakistan": (30.0, 70.0),
    "Philippines": (13.0, 122.0),
    "Indonesia": (-0.8, 113.0),
    "Iran": (32.0, 53.0),
    "Iraq": (33.0, 44.0),
    "Saudi Arabia": (25.0, 45.0),
    "South Africa": (-29.0, 24.0),
    "Egypt": (27.0, 30.0),
    "Tunisia": (33.5, 9.0),
    "Morocco": (32.0, -5.0),
    "Algeria": (28.0, 1.0),
    "Norway": (62.0, 10.0),
    "Sweden": (60.0, 15.0),
    "Finland": (64.0, 26.0),
    "Denmark": (56.0, 10.0),
    "Poland": (52.0, 19.0),
    "Czech Republic": (49.7, 15.5),
    "Hungary": (47.0, 19.5),
    "Romania": (46.0, 25.0),
    "Bulgaria": (43.0, 25.0),
    "Cyprus": (35.0, 33.0),
}
# Region -> country override. Lets "Western Sicily" or "Sicilian
# sections" map to "Italy" without listing Sicily (a region, not a
# country) in the main country list. Add new region->country pairs
# here when papers consistently use a regional name.
_REGION_TO_COUNTRY = (
    ("sicily", "Italy"),
    ("sicilian", "Italy"),
    ("tuscany", "Italy"),
    ("sardinia", "Italy"),
    ("calabria", "Italy"),
    ("andalusia", "Spain"),
    ("bohemia", "Czech Republic"),
    ("scotland", "United Kingdom"),
    ("england", "United Kingdom"),
    ("wales", "United Kingdom"),
    ("hokkaido", "Japan"),
    ("honshu", "Japan"),
    ("shikoku", "Japan"),
    ("kyushu", "Japan"),
)
_COUNTRY_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in _COUNTRIES) + r")\b"
)

# Modern vs paleo coordinate heuristic. A "paleo" coordinate appears
# in a sentence that frames it as the position AT DEPOSITION TIME
# (e.g. "the basin was located at 23°S, 47°W in the Late Triassic").
# A "modern" coordinate appears in a present-day framing ("today
# the locality is at 38°N, 14°E"). Keywords that flip the assignment.
_PALEO_KEYWORDS = (
    "during the ", "at that time", "at the time", "in the late ",
    "in the early ", "in the middle ", "paleogeographic",
    "paleolatitude", "paleolongitude", "during deposition",
    "reconstructed", "was located", "lay at", "was situated",
    "at deposition", "in triassic", "in jurassic", "in cretaceous",
    "in permian", "in devonian", "in ordovician", "in silurian",
    "in cambrian", "in carboniferous",
    # Phase 62 Plan 5 (Bug 5.15): era + epoch names. The original
    # list covered periods only, so a sentence framed "in the
    # Eocene" / "in the Mesozoic" was mis-classified as modern.
    # Both "in the X" and bare "X" framings are covered — real
    # sentences use both forms interchangeably.
    "in mesozoic", "in the mesozoic", "mesozoic",
    "in cenozoic", "in the cenozoic", "cenozoic",
    "in paleozoic", "in the paleozoic", "paleozoic",
    "in paleogene", "in the paleogene", "paleogene",
    "in neogene", "in the neogene", "neogene",
    "in eocene", "in the eocene", "eocene",
    "in oligocene", "in the oligocene", "oligocene",
    "in miocene", "in the miocene", "miocene",
    "in pliocene", "in the pliocene", "pliocene",
    "in pleistocene", "in the pleistocene", "pleistocene",
)
_MODERN_KEYWORDS = (
    "today", "present-day", "present day", "currently",
    "now ", "modern coordinates", "modern position",
    "modern locality",
)


def _strip_leading_article(name: str | None) -> str | None:
    """Strip leading English articles ("The ", "A ", "An ") from a
    stratigraphic unit name so "The Lower Member" becomes "Lower
    Member". Returns the original value if ``None``.
    """
    if not name:
        return name
    for art in ("The ", "the ", "A ", "a ", "An ", "an "):
        if name.startswith(art):
            return name[len(art):]
    return name


# Phase 62 Plan 5 (Bug 5.5): stopword prefixes the article-strip
# helper above does NOT handle. Real formation/group/member names
# never begin with these; a match that does is almost always the
# regex catching a sentence-spanning phrase ("In Group we infer
# age" → "In Group") rather than a real unit name. Used by
# ``_formation_name_ok`` to reject the match.
_FORMATION_STOPWORD_PREFIXES = (
    "the ", "The ",
    "a ", "A ",
    "an ", "An ",
    "of ", "Of ",
    "in ", "In ",
    "and ", "And ",
    "from ", "From ",
    "near ", "Near ",
    "by ", "By ",
)


def _starts_with_stopword(name: str | None) -> bool:
    """Return True if ``name`` (after the trailing formation/group/member
    keyword has been stripped) begins with a stopword prefix.

    Checks both forms:
      * ``"In " + space`` — the prefix is followed by other words.
      * ``"In"`` alone — the prefix is the entire stripped name
        (e.g. "In Group" stripped → "In").
    """
    if not name:
        return False
    # Match with trailing space (multi-word name starting with a
    # stopword: "In Sicanian Group" stripped → "In Sicanian").
    if any(name.startswith(p) for p in _FORMATION_STOPWORD_PREFIXES):
        return True
    # Match when the entire stripped name IS the stopword.
    first_word = name.split(" ", 1)[0].lower()
    stopwords = {p.strip().lower() for p in _FORMATION_STOPWORD_PREFIXES}
    return first_word in stopwords


def _classify_coordinate_age(
    text: str, match_start: int, match_end: int
) -> str:
    """Return ``"paleo"`` or ``"modern"`` based on keywords within
    ~120 chars BEFORE the coordinate match. Defaults to ``"modern"``
    when neither set of keywords appears (most radiolarian papers
    report modern locality coordinates by default).
    """
    ctx = text[max(0, match_start - 120) : match_start].lower()
    for kw in _PALEO_KEYWORDS:
        if kw in ctx:
            return "paleo"
    for kw in _MODERN_KEYWORDS:
        if kw in ctx:
            return "modern"
    return "modern"


@dataclass(slots=True)
class GeologyRecord:
    age: str | None = None  # period name (e.g. "Permian")
    chronostratigraphy: str | None = None  # most specific stage (e.g. "Changhsingian")
    chronostratigraphy_rank: str | None = None  # "period" | "epoch" | "age"
    # Numeric Ma bounds (top = younger, base = older; ICS convention).
    # Populated from AgeClassification.ma_top / ma_base / ma_mid which
    # is itself derived from the matched ICS row. Until this commit
    # these fields were always None because the producer chain dropped
    # the Ma values; the converter faithfully read them, so wiring
    # this up at the source end of the chain lights them up in the
    # exported GeologyLinkRecord / GeologyContextRecord automatically.
    ma_top: float | None = None
    ma_base: float | None = None
    ma_mid: float | None = None
    formation: str | None = None
    # Round 18 audit: split the formation regex output so a paper
    # that mentions all three ranks (Group → Formation → Member)
    # emits three separate fields. Previously everything landed in
    # ``formation`` only.
    group: str | None = None
    member: str | None = None
    lithology: str | None = None  # e.g. "siliceous limestone"
    locality: str | None = None
    country: str | None = None
    biozone: str | None = None  # e.g. "N. optima Zone"
    latitude: float | None = None
    longitude: float | None = None
    # Round 18 audit: split lat/lon into modern vs paleo so the
    # converter downstream can keep them distinct. The plain
    # ``latitude``/``longitude`` keep the first coord found (which
    # is usually modern for radiolarian locality tables).
    modern_latitude: float | None = None
    modern_longitude: float | None = None
    paleo_latitude: float | None = None
    paleo_longitude: float | None = None
    section_type: str | None = None
    section_title: str | None = None
    evidence_text: str | None = None
    confidence: float = 0.0
    # Round 21: how the coordinate was derived. Empty for regex-
    # extracted coords (the default path), ``"country_centroid"``
    # when the country-centroid fallback table supplied the
    # latitude/longitude. The ``converters`` path doesn't currently
    # surface this field, but it's available for downstream
    # inspection (e.g. UI badge "derived from country centroid").
    coord_source: str = ""
    # Round 24: environment / geochem proxies. See the schema
    # docstrings for the field semantics. Populated by
    # ``extract_geology_from_sections`` from the curated
    # vocabularies.
    paleoenvironment: str | None = None
    redox: str | None = None
    chemostrat: str | None = None
    facies: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_geology_from_sections(sections: list[dict[str, str]]) -> list[GeologyRecord]:
    out: list[GeologyRecord] = []
    # Lazy import to avoid circular
    try:
        from .stratigraphy import classify_age_string, find_ages_in_text
    except Exception:
        find_ages_in_text = None
        classify_age_string = None
    for sec in sections:
        text = sec.get("text", "")
        if not text:
            continue
        # Round 20: skip references / bibliography sections entirely.
        # Citation paragraphs mention formation names, countries, and
        # localities as part of bibliographic titles — they are
        # NOT geology facts of THIS paper. The Danelian 2006 paper
        # leak (country=Japan, formation="Fonzaso Formation") was
        # traced to a Beccaro 2002 reference being extracted as if
        # it were Danelian's own geology. References must be
        # filtered out at this point so no downstream code can
        # confuse them with the paper's actual stratigraphy.
        sec_type = (sec.get("section_type") or "").lower()
        sec_title = (sec.get("title") or "").lower()
        if sec_type == "references" or any(
            kw in sec_title
            for kw in ("reference", "bibliograph", "cited work", "literature cited")
        ):
            logger.debug(
                "Skipping section %r (type=%r): bibliography / references section "
                "cannot be a source of geology facts",
                sec.get("title"),
                sec_type,
            )
            continue
        # Round 20: validate each AGE_PATTERN match against the ICS
        # stratigraphy lexicon. The raw regex would match anything
        # that looks like "Late <Capitalized>" or a period name; in
        # real text, phrases like "lower part of the section",
        # "upper reaches of the formation", or "Late effects of
        # diagenesis" all match the first alternative and were
        # leaking into the ``age`` column of GeologyLinkRecord. We
        # classify each match through ``classify_age_string`` and
        # keep only those that the lexicon recognises (confidence
        # > 0). Empty / unrecognised ages are dropped, not stored
        # as literal strings — geologists cannot interpret
        # ``age="lower part"`` meaningfully.
        raw_age_matches = [m.group(0).strip() for m in AGE_PATTERN.finditer(text)]
        if classify_age_string is not None:
            ages: list[str] = []
            for raw in raw_age_matches:
                cls = classify_age_string(raw)
                if cls.confidence > 0 and (cls.period or cls.epoch or cls.age):
                    ages.append(raw)
                else:
                    logger.debug(
                        "AGE_PATTERN matched %r but stratigraphy lexicon "
                        "rejected it (confidence=%.2f); dropping",
                        raw,
                        cls.confidence,
                    )
        else:
            ages = raw_age_matches
        # Round 20: filter locality matches against the stratigraphy
        # lexicon. ``LOCALITY_PATTERN`` (e.g. ``from <Capitalized>``)
        # over-matches when the preposition "from" / "in" / "at"
        # precedes a known stratigraphic term. The Bandini 2006
        # SPECIES LIST section starts with "Acaeniotyle ... from
        # Upper Cretaceous formations" — the phrase "from Upper
        # Cretaceous" was captured as locality=``"Upper Cretaceous"``
        # even though it's a stratigraphic age, not a place. We
        # reject any locality that the ICS lexicon recognises as a
        # real period / epoch / age.
        raw_locs = [m.group(1).strip(" .,;") for m in LOCALITY_PATTERN.finditer(text)]
        locs: list[str] = []
        for loc in raw_locs:
            if classify_age_string is not None:
                cls = classify_age_string(loc)
                if cls.confidence > 0 and (cls.period or cls.epoch or cls.age):
                    logger.debug(
                        "LOCALITY_PATTERN matched %r but stratigraphy "
                        "lexicon recognised it as %s; treating as age, "
                        "not locality",
                        loc,
                        cls.rank,
                    )
                    continue
            locs.append(loc)
        # Round 18: split the formation regex into rank-specific
        # matches so a single text span yields separate
        # ``group`` / ``formation`` / ``member`` fields instead of
        # collapsing all three into ``formation``. The legacy greedy
        # FORMATION_PATTERN was retired — it matched across sentence
        # boundaries (e.g. "The Sicanian Group contains the Lower
        # Member which is siliceous limestone" swallowed the whole
        # sentence as ``formation``).
        #
        # Round 20 hardening: post-filter to drop any match whose
        # name prefix contains digits. The regex uses
        # ``[A-Z][A-Za-z\-]{0,30}?`` which can absorb tokens like
        # "19" / "20" in page references or sample IDs
        # ("Karnezeika-19 Formation", "Bed 5 Formation"). Real
        # formation names never start with a digit in the first
        # capitalised token, so a digit anywhere in the prefix is a
        # strong signal that the match is misaligned with the
        # "Formation/Fm." keyword that follows.
        def _formation_name_ok(name: str) -> bool:
            # Strip the trailing "Formation"/"Fm." and check the
            # remaining name. The trailing keyword is removed first
            # so its presence doesn't pollute the digit check.
            stripped = name
            for kw in ("Formation", "Fm.", "Group", "Gp.", "Member", "Mb."):
                if stripped.endswith(kw):
                    stripped = stripped[: -len(kw)].rstrip()
                    break
            # Phase 62 Plan 5 (Bug 5.5): reject stopword prefixes
            # (The, A, An, Of, In, And, From, Near, By). Real
            # formation/group/member names never begin with these;
            # a match that does is almost always the regex catching
            # a sentence-spanning phrase.
            if _starts_with_stopword(stripped):
                return False
            return not any(ch.isdigit() for ch in stripped)

        groups = [
            m.group(1).strip()
            for m in _GROUP_RE.finditer(text)
            if _formation_name_ok(m.group(1))
        ]
        formations = [
            m.group(1).strip()
            for m in _FORMATION_RE.finditer(text)
            if _formation_name_ok(m.group(1))
        ]
        members = [
            m.group(1).strip()
            for m in _MEMBER_RE.finditer(text)
            if _formation_name_ok(m.group(1))
        ]
        # ``forms`` is only used as a "did we find ANY formation
        # keyword at all" gate in the skip-check below. We use the
        # union of the three rank lists so we don't drop records that
        # only mentioned "Group" or "Member".
        forms = formations or groups or members
        # Lithology: pick the first match (dedup case-insensitive).
        lithology = None
        seen_litho: set[str] = set()
        for m in LITHOLOGY_PATTERN.finditer(text):
            lit = m.group(0).strip()
            # Round 23 fix: rename ``key`` → ``lit_key`` so mypy can
            # disambiguate from the later ``key`` tuple used for
            # ``rec not in out`` dedup. The previous name shadowed
            # the outer-scope tuple and tripped the
            # ``assignment has type tuple / variable has type str``
            # error on line 622.
            lit_key = lit.lower()
            if lit_key in seen_litho:
                continue
            seen_litho.add(lit_key)
            lithology = lit
            break
        # Biozone: pick first match
        bio_match = _BIOZONE_RE.search(text)
        biozone = bio_match.group(0).strip() if bio_match else None
        # Round 24: paleoenvironment / redox / chemostrat / facies
        # proxies. Each is matched case-insensitive whole-word.
        # First match wins (the operator cares about the dominant
        # signal, not every mention). All four are optional — the
        # section is not penalised for missing these.
        paleoenv_match = PALEOENV_PATTERN.search(text)
        paleoenvironment = paleoenv_match.group(0).strip() if paleoenv_match else None
        redox_match = REDOX_PATTERN.search(text)
        redox = redox_match.group(0).strip() if redox_match else None
        chemostrat_match = CHEMOSTRAT_PATTERN.search(text)
        chemostrat = chemostrat_match.group(0).strip() if chemostrat_match else None
        facies_match = FACIES_PATTERN.search(text)
        facies = facies_match.group(0).strip() if facies_match else None
        # Round 25: isotope values (δ13C / δ18O / 87Sr/86Sr / TOC
        # / Hg). The user audit specifically called out "geochem
        # values" as a missing data point for P/T boundary
        # research. We capture the FIRST match per section and
        # append it to evidence_text so the operator can see the
        # exact value without re-parsing the paper.
        iso_match = _ISOTOPE_PATTERN.search(text)
        isotope_value = iso_match.group(0).strip() if iso_match else None
        # Country: search anywhere in the section text. Order
        # matters only when a paper names both a country and a
        # sub-region locality — we keep the first match in the text.
        country_match = _COUNTRY_RE.search(text)
        country = country_match.group(1) if country_match else None
        # Round 18: if a region name (e.g. "Sicily") is mentioned
        # but no country, fall back to the region->country override
        # table so the operator still gets a country on a Sicily /
        # Tuscany / Bohemia paper. The check is whole-word and
        # case-insensitive on the lowercased text.
        if country is None:
            text_l = text.lower()
            for region, fallback_country in _REGION_TO_COUNTRY:
                if f" {region} " in f" {text_l} ":
                    country = fallback_country
                    break

        # Stratigraphy enrichment — find stage names (Changhsingian, Wuchiapingian, …)
        chrono = None
        chrono_rank = None
        ma_top = ma_base = ma_mid = None
        if find_ages_in_text is not None:
            try:
                cls_list = find_ages_in_text(text)
                # Pick the most specific one (rank="age" > "epoch" > "period")
                rank_order = {"age": 3, "epoch": 2, "period": 1}
                best = max(
                    [c for c in cls_list if c.confidence > 0],
                    key=lambda c: rank_order.get(c.rank or "", 0),
                    default=None,
                )
                if best is not None:
                    chrono = best.age or best.period
                    chrono_rank = best.rank
                    # Carry numeric Ma values from the matched ICS row.
                    # These flow through to_dict() -> converters and
                    # reach the published GeologyLinkRecord.ma_* fields.
                    ma_top = best.ma_top
                    ma_base = best.ma_base
                    ma_mid = best.ma_mid
            except Exception:
                pass
        # Coordinate parsing. ``_extract_first_coord`` already validates
        # the latitude/longitude ranges (rejects e.g. lat=200, lon=400)
        # and applies hemisphere flips. The previous code re-ran the
        # regex and overwrote lat/lon with values that BYPASSED the
        # range check, which let invalid coordinates leak into
        # GeologyRecord.latitude/longitude.
        lat, lon = _extract_first_coord(text)
        # Round 21: country-centroid fallback. When the section text
        # has no explicit coordinates (only a country name like
        # "Greece" or "Tunisia"), look up the centroid as a low-
        # confidence fallback. The ``confidence`` flag on the
        # GeologyRecord reflects this: regex-extracted coords are
        # 0.7 / 0.55; centroid fallback is 0.3 (and the
        # ``coord_source`` field carries "country_centroid" so the
        # operator can tell).
        centroid_source = ""
        if lat is None and lon is None and country is not None:
            centroid = _COUNTRY_CENTROIDS.get(country)
            if centroid is not None:
                lat, lon = centroid
                centroid_source = "country_centroid"
                logger.debug(
                    "country centroid fallback: country=%r → lat=%s lon=%s",
                    country,
                    lat,
                    lon,
                )
        # Round 18: classify the coordinate as paleo vs modern based
        # on surrounding keywords ("at deposition time" → paleo,
        # "today / present-day" → modern). Without this, both
        # modern_latitude and paleo_latitude hold the same value
        # and the user can't tell which is which.
        if lat is not None and lon is not None:
            coord_match = COORDINATE_PATTERN.search(text)
            if coord_match is not None:
                coord_age = _classify_coordinate_age(
                    text, coord_match.start(), coord_match.end()
                )
            else:
                coord_age = "modern"
        else:
            coord_age = "modern"

        if (
            not ages
            and not forms
            and not locs
            and chrono is None
            and lat is None
            and not groups
            and not formations
            and not members
            and lithology is None
            and biozone is None
            and country is None
            # Round 24+25: also pass the skip-check if the new
            # proxies are populated. Otherwise a record with ONLY
            # chemostrat / paleoenvironment / redox / facies would
            # be silently dropped. The proxies are the entire point
            # of the new extraction so a record with no other
            # fields but a useful proxy is still valuable.
            and paleoenvironment is None
            and redox is None
            and chemostrat is None
            and facies is None
        ):
            continue

        # 以句子级片段做证据，先走规则抽取。
        # If we have chrono from stratigraphy, use that as age.
        primary_age = chrono if chrono else (ages[0] if ages else None)
        for age in ages or [None]:
            rec = GeologyRecord(
                age=age or primary_age,
                chronostratigraphy=chrono,
                chronostratigraphy_rank=chrono_rank,
                ma_top=ma_top,
                ma_base=ma_base,
                ma_mid=ma_mid,
                group=_strip_leading_article(groups[0]) if groups else None,
                formation=_strip_leading_article(formations[0]) if formations else None,
                member=_strip_leading_article(members[0]) if members else None,
                lithology=lithology,
                # Round 24: environment / geochem proxies.
                paleoenvironment=paleoenvironment,
                redox=redox,
                chemostrat=chemostrat,
                facies=facies,
                locality=locs[0] if locs else None,
                country=country,
                biozone=biozone,
                latitude=lat,
                longitude=lon,
                modern_latitude=lat if coord_age == "modern" else None,
                modern_longitude=lon if coord_age == "modern" else None,
                paleo_latitude=lat if coord_age == "paleo" else None,
                paleo_longitude=lon if coord_age == "paleo" else None,
                section_type=sec.get("section_type"),
                section_title=sec.get("title"),
                # Round 25: append the captured isotope value to
                # the evidence text so the operator can see
                # "δ13C = -3.2 ‰" without re-parsing the paper.
                # Truncated to 300 chars to keep the cell small.
                evidence_text=(
                    (text + (f"  [isotope: {isotope_value}]" if isotope_value else ""))[:300]
                ),
                # Round 21: country-centroid fallback coords are
                # tagged with lower confidence (0.3) so the
                # downstream UI / consumers can distinguish them
                # from regex-extracted coords. Regex-extracted
                # values stay at 0.7 / 0.55.
                confidence=0.3 if centroid_source else (0.7 if chrono else 0.55),
                coord_source=centroid_source,
            )
            out.append(rec)
        # If no ages were found but chrono was, still emit a record
        if not ages and chrono:
            rec = GeologyRecord(
                age=primary_age,
                chronostratigraphy=chrono,
                chronostratigraphy_rank=chrono_rank,
                ma_top=ma_top,
                ma_base=ma_base,
                ma_mid=ma_mid,
                group=_strip_leading_article(groups[0]) if groups else None,
                formation=_strip_leading_article(formations[0]) if formations else None,
                member=_strip_leading_article(members[0]) if members else None,
                lithology=lithology,
                # Round 24: environment / geochem proxies.
                paleoenvironment=paleoenvironment,
                redox=redox,
                chemostrat=chemostrat,
                facies=facies,
                locality=locs[0] if locs else None,
                country=country,
                biozone=biozone,
                latitude=lat,
                longitude=lon,
                modern_latitude=lat if coord_age == "modern" else None,
                modern_longitude=lon if coord_age == "modern" else None,
                paleo_latitude=lat if coord_age == "paleo" else None,
                paleo_longitude=lon if coord_age == "paleo" else None,
                section_type=sec.get("section_type"),
                section_title=sec.get("title"),
                # Round 25: append the captured isotope value to
                # the evidence text so the operator can see
                # "δ13C = -3.2 ‰" without re-parsing the paper.
                # Truncated to 300 chars to keep the cell small.
                evidence_text=(
                    (text + (f"  [isotope: {isotope_value}]" if isotope_value else ""))[:300]
                ),
                confidence=0.3 if centroid_source else 0.7,
                coord_source=centroid_source,
            )
            # Compare on a stable key tuple instead of the full dataclass.
            # `dataclass(slots=True)` gives an auto-generated __eq__ that
            # works today, but ``rec not in out`` does linear scans and
            # would silently fail (always True) if a future field was
            # excluded from eq=. The key tuple is the same one
            # ``dedup_geology_records`` uses below, so the two paths are
            # consistent.
            key = (
                rec.age,
                rec.chronostratigraphy,
                rec.formation,
                rec.locality,
                rec.section_title,
            )
            existing_keys = {
                (r.age, r.chronostratigraphy, r.formation, r.locality, r.section_title) for r in out
            }
            if key not in existing_keys:
                out.append(rec)
    return dedup_geology_records(out)


def _extract_first_coord(text: str) -> tuple[float | None, float | None]:
    """Best-effort coordinate extraction. Returns ``(lat, lon)`` or ``(None, None)``.

    Requires at least one hemisphere indicator (N/S/E/W) or a degree symbol
    (°) in the match. Without this filter, bare number pairs like "45, 90"
    (page numbers, specimen dimensions, figure counts) would be falsely
    parsed as coordinates.
    """
    if not text:
        return None, None
    for m in COORDINATE_PATTERN.finditer(text):
        # Require at least one hemisphere indicator or degree symbol to
        # reduce false positives from bare number pairs.
        has_hemisphere = bool(m.group(2) or m.group(4))
        has_degree = "°" in m.group(0)
        if not has_hemisphere and not has_degree:
            continue
        try:
            lat = float(m.group(1))
            if m.group(2) and m.group(2).upper() == "S":
                lat = -lat
            lon = float(m.group(3))
            if m.group(4) and m.group(4).upper() == "W":
                lon = -lon
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue
            return lat, lon
        except Exception:
            continue
    return None, None


def link_species_to_geology(
    species_names: list[str],
    sections: list[dict[str, str]],
    llm_runtime: Any | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Link species to geology records.
    If llm_runtime is provided, use LLM for relation refinement; else use proximity heuristics.

    Round 20 sampling: Danelian 2006 panel 6 leaked
    country=Japan / formation="Fonzaso Formation" through this
    function. The "Systematic Palaeontology" section discusses
    individual species and frequently cites other papers' type
    localities / formations in synonymy lists — those citations
    are NOT Danelian's own geology. To prevent the leak, this
    function filters ``sections`` to only those typed as
    ``geological_setting`` (or any non-excluded type if no
    ``geological_setting`` section exists). The same filtering
    was applied to ``link_panels_to_geology`` in Round 20.
    """
    # Filter sections to trust-worthy geology sources. Same logic
    # as ``link_panels_to_geology``: ``geological_setting`` first,
    # then everything except ``references`` and
    # ``systematic_paleontology`` as a fallback for papers that
    # don't use typed sections.
    geology_section_types = {"geological_setting"}
    geo_sections = [
        sec
        for sec in sections
        if (sec.get("section_type") or "").lower() in geology_section_types
    ]
    if not geo_sections:
        geo_sections = [
            sec
            for sec in sections
            if (sec.get("section_type") or "").lower()
            not in {"references", "systematic_paleontology"}
        ]
    geology = extract_geology_from_sections(geo_sections)
    links: dict[str, list[dict[str, Any]]] = {s: [] for s in species_names}
    if not species_names:
        return links

    if llm_runtime is None:
        # Heuristic linking: a species is linked to a geology record
        # ONLY when the species name actually appears in the same
        # section as the record. The previous implementation also
        # appended a fallback record to every unmatched species,
        # which produced the same 5-record dump for every panel of
        # a paper that uses generic any-age captions. The new
        # behaviour is honest: if the operator has no LLM and no
        # OCR, a panel with no detectable species ends up with NO
        # geology links rather than fabricated ones.
        for s in species_names:
            s_lower = s.lower()
            for sec in sections:
                text = (sec.get("text") or "").lower()
                if s_lower in text:
                    for rec in geology:
                        if rec.section_title == sec.get("title"):
                            links[s].append(rec.to_dict())
        return links

    # 使用 LLM 关系链接。
    from .gemma_postprocess import gemma_extract_text_json

    system_prompt = (
        "You are a scientific IE assistant. Given species name and section text, "
        "extract linked geology fields as strict JSON with keys: label,species,confidence,reasoning."
    )
    for s in species_names:
        best_records: list[dict[str, Any]] = []
        for sec in geo_sections:
            text = sec.get("text", "")
            if not text:
                continue
            user_prompt = (
                f"Species: {s}\nSection title: {sec.get('title')}\n"
                f"Section type: {sec.get('section_type')}\n"
                f"Text: {text[:1500]}\n"
                'Return JSON: {"label":"geo_link","species":"...","confidence":0-1,"reasoning":"age=...,formation=...,locality=..."}'
            )
            out = gemma_extract_text_json(llm_runtime, system_prompt, user_prompt)
            conf = float(out.get("confidence", 0.0))
            if conf < 0.4:
                continue
            reasoning = str(out.get("reasoning", ""))
            # Round 25: the LLM path doesn't run the isotope
            # regex on the LLM response (the regex runs on the
            # raw section text in the regex path). Default to
            # None so the f-string is well-formed.
            iso_match = _ISOTOPE_PATTERN.search(reasoning)
            isotope_value = iso_match.group(0).strip() if iso_match else None
            rec = GeologyRecord(
                age=_extract_first(AGE_PATTERN, reasoning),
                formation=_extract_first(FORMATION_PATTERN, reasoning),
                locality=_extract_first(LOCALITY_PATTERN, reasoning),
                section_type=sec.get("section_type"),
                section_title=sec.get("title"),
                # Round 25: append the captured isotope value to
                # the evidence text so the operator can see
                # "δ13C = -3.2 ‰" without re-parsing the paper.
                # Truncated to 300 chars to keep the cell small.
                evidence_text=(
                    (text + (f"  [isotope: {isotope_value}]" if isotope_value else ""))[:300]
                ),
                confidence=conf,
            )
            best_records.append(rec.to_dict())
        links[s] = best_records[:5]
    return links


def link_panels_to_geology(
    captions: dict[str, str],
    fallback_sections: list[dict[str, str]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Link each panel to the geology facts MENTIONED IN ITS OWN CAPTION.

    This is the panel-scoped counterpart of
    :func:`link_species_to_geology`. Where the species-level version
    asks "which species lives in which section", this function asks
    "which age/formation/locality does THIS panel's caption mention".

    Parameters
    ----------
    captions : dict[str, str]
        Mapping ``panel_id -> caption_text``. The panel_id is free-form
        (e.g. ``"auto_fig_p004_r01:panel_01"``) and is preserved in the
        returned dict so the caller can wire it back into PanelRecord.
    fallback_sections : list[dict[str, str]] | None
        Optional list of fulltext sections, used as a SOURCE of geology
        records (NOT as the search text). The previous implementation
        searched fulltext for the panel's caption; this version keeps
        the search local to the caption and only pulls the candidate
        records (with their proper evidence_text) from the fulltext.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        ``{panel_id: [geology_record_dict, ...]}``. Panels whose
        caption does not mention any age/formation/locality get an
        empty list -- this is intentional and is the fix for the
        "every panel inherits the whole paper's geology" bug.
    """
    # Build the candidate record pool from the fulltext once. The
    # fulltext is used to populate `evidence_text` and the section
    # metadata, but the AGE/FORMATION/LOCALITY facts must appear in
    # the panel's own caption -- not in some other paragraph of
    # the body text.
    #
    # Round 20 sampling: Danelian 2006 panel 6 was getting
    # country=Japan / formation="Fonzaso Formation" from the
    # "Systematic Palaeontology" section. That section discusses
    # individual species and frequently cites OTHER papers' type
    # localities / formations in synonymy lists ("previously
    # reported from Japan", "Beccaro 2002 on Fonzaso Formation").
    # Those citations are NOT the paper's own geology. To prevent
    # the leak, only pull candidates from sections that actually
    # describe the paper's own stratigraphy / locality / age.
    candidates: list[GeologyRecord] = []
    if fallback_sections:
        # Section types we trust as geology sources for the
        # candidate pool. ``systematic_paleontology`` /
        # ``materials_methods`` / ``other`` are excluded because
        # they discuss individual species (with citation leakage)
        # or methods, not the paper's own geology.
        geology_section_types = {"geological_setting"}
        geo_sections = [
            sec
            for sec in fallback_sections
            if (sec.get("section_type") or "").lower() in geology_section_types
        ]
        # If no explicit geological_setting section exists, fall
        # back to the union of all sections EXCEPT references and
        # systematic_paleontology — preserving the Round 19
        # behaviour for papers that don't use the typed sections.
        if not geo_sections:
            geo_sections = [
                sec
                for sec in fallback_sections
                if (sec.get("section_type") or "").lower()
                not in {"references", "systematic_paleontology"}
            ]
        candidates = extract_geology_from_sections(geo_sections)
    out: dict[str, list[dict[str, Any]]] = {}
    # Memoise extraction results per unique caption text. In the
    # typical case (all panels in one figure), every panel_id maps
    # to the same caption string, so we only need to run the regex
    # pipeline once rather than N times (once per panel). The
    # cache key is the raw caption string; panel_id is not used
    # inside ``extract_geology_from_sections`` so two calls with
    # the same caption always produce the same records.
    _cap_cache: dict[str, list[GeologyRecord]] = {}
    for panel_id, caption in captions.items():
        if not caption:
            out[panel_id] = []
            continue
        records: list[GeologyRecord] = []
        # 1) Always try to extract FROM the caption first -- the
        # panel's own text is the authoritative source.
        # Use a per-caption cache so the regex pipeline runs once
        # per unique caption string rather than once per panel
        # (panels in the same figure share the same caption text;
        # the panel_id is cosmetic and does not affect extraction).
        if caption not in _cap_cache:
            _cap_cache[caption] = extract_geology_from_sections(
                [{"title": f"panel:{panel_id}", "text": caption, "section_type": "panel_caption"}]
            )
        # Deep-copy so per-panel metadata tweaks (confidence,
        # evidence_text) don't pollute the cached originals.
        for r in _cap_cache[caption]:
            r2 = replace(r)
            r2.confidence = max(r2.confidence, 0.6)
            r2.evidence_text = caption[:300]
            records.append(r2)
        # 2) If the caption is short ("Auto-generated figure for
        # page 1") or matches nothing, fall back to the
        # fulltext-derived candidates BUT only for facts the
        # caption actually mentions. This is the safety net
        # for papers where the caption is non-informative but
        # the body text mentions the same age the plate is
        # illustrating.
        if not records and candidates:
            cap_l = caption.lower()
            for c in candidates:
                # A candidate is "echoed in the caption" only when
                # at least one of its key fields is a substring.
                keys = (c.age, c.formation, c.locality, c.chronostratigraphy)
                if any(k and k.lower() in cap_l for k in keys):
                    records.append(c)
        # 3) If the caption is a known placeholder string (very
        # short, contains "auto-generated" / "placeholder" /
        # "page N") AND we have fulltext sections, attach the
        # most relevant fulltext geology record by section
        # title. We pick the FIRST section (typically the
        # "Geological setting" or "Introduction" section that
        # discusses the paper's stratigraphy and locality) --
        # this gives the operator a sensible per-paper default
        # without dumping every age in the paper onto every
        # panel.
        if not records and candidates and _is_placeholder_caption(caption):
            # Pick candidates whose section_title is the FIRST
            # one in the fulltext. This anchors all panels in
            # the paper to the same introductory stratigraphy
            # context, which is the most common real-world case.
            first_title = fallback_sections[0].get("title") if fallback_sections else None
            for c in candidates:
                if first_title and c.section_title == first_title:
                    records.append(c)
            # If no candidates matched the first section (e.g.
            # the fulltext had no usable age mentions), fall
            # back to the first candidate. This is the only
            # path that uses a "global default" -- the previous
            # implementation applied this to every panel, the
            # new code applies it only to placeholders.
            if not records and candidates:
                records.append(candidates[0])
        out[panel_id] = [r.to_dict() for r in records]
    return out


def build_knowledge_graph(links: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[tuple[str, str]] = set()
    for species, records in links.items():
        species_node = ("species", species)
        if species_node not in seen_nodes:
            seen_nodes.add(species_node)
            nodes.append({"id": f"species:{species}", "type": "species", "name": species})

        for idx, rec in enumerate(records):
            for field in ("age", "formation", "locality"):
                value = rec.get(field)
                if not value:
                    continue
                node_key = (field, value)
                if node_key not in seen_nodes:
                    seen_nodes.add(node_key)
                    nodes.append({"id": f"{field}:{value}", "type": field, "name": value})
                edges.append(
                    {
                        "source": f"species:{species}",
                        "target": f"{field}:{value}",
                        "relation": f"has_{field}",
                        "confidence": rec.get("confidence", 0.0),
                    }
                )
    return {"nodes": nodes, "edges": edges}


def dedup_geology_records(records: list[GeologyRecord]) -> list[GeologyRecord]:
    out: dict[tuple[Any, ...], GeologyRecord] = {}
    for rec in records:
        key = (rec.age, rec.chronostratigraphy, rec.formation, rec.locality, rec.section_title)
        old = out.get(key)
        if old is None or rec.confidence > old.confidence:
            out[key] = rec
    return list(out.values())


def _extract_first(pattern: re.Pattern, text: str) -> str | None:
    m = pattern.search(text or "")
    if not m:
        return None
    if m.lastindex:
        return str(m.group(1)).strip()
    return str(m.group(0)).strip()


# Inline placeholder-caption detector. Mirrors
# rlpe.text_filters.looks_like_placeholder_caption. We delegate to
# text_filters as the single source of truth so the two paths cannot
# drift (the previous local regex required a digit after "page" while
# text_filters did not, so a caption like "page auto-generated"
# matched here but not there). Keep the local function name so
# existing callers don't change.
from .text_filters import (
    looks_like_placeholder_caption as _is_placeholder_caption,  # noqa: E402,F401
)
