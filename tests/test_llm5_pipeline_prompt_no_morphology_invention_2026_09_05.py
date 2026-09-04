"""Regression: audit 2026-09-04 llm-5 — the LLM-first system
prompt in :class:`rlpe.pipeline.RLPipeline` told the model to
NAME species from the image morphology using parametric knowledge
when the caption didn't already name them:

    src/rlpe/pipeline.py:5057
        - SECOND: if the caption does NOT mention species for a
          panel, try to identify the species from the image
          morphology using your knowledge of radiolarian taxonomy.
          Set confidence lower (0.3-0.5) to indicate this is a
          morphology-based guess, not a caption-confirmed
          identification.

The model is NOT a radiolarian expert. The "guess from morphology"
rule was the root cause of fabricated species names in M3 outputs
— the model invented binomial strings that look plausible but
don't correspond to a real species. These fabricated names then
flowed into Darwin Core exports and GBIF submissions.

Real failure mode (Round 6 live): a Bandini 2011 plate caption
lists species for panels 1-5 but NOT for panel 6. The model
guessed a name from morphology, emitted ``{"species": "Genus
hallucinata", "confidence": 0.4}``. The downstream DwC export
recorded this as a valid observation.

Fix contract:
  * Remove the "identify from morphology using your knowledge"
    rule from the LLM-first system prompt.
  * Replace it with: "If the caption does NOT mention species for
    a panel, set ``species: null``. Do not guess from morphology.
    Radiolarian taxonomy identification requires expert
    reference; an LLM is not one."
  * The "NEVER invent species names" rule stays (it's correct,
    just needs to be enforced by removing the conflicting
    "morphology guess" rule).

This test pins the fix via source guards over the LLM-first
system prompt (the prompt is a class constant on RLPipeline).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
sys.path.insert(0, str(_SRC))


class TestLLMFirstPromptForbidsMorphologyGuess:
    def test_no_morphology_guess_rule(self):
        """The prompt must not instruct the model to NAME species
        from image morphology when the caption doesn't mention
        them."""
        from rlpe.pipeline import RadiolarianPipeline

        prompt = RadiolarianPipeline._LLM_FIRST_SYSTEM_PROMPT
        prompt_lower = prompt.lower()
        # Forbidden phrasings: anything that tells the model to
        # use its knowledge of taxonomy / morphology to name a
        # species the caption didn't provide.
        forbidden_phrases = [
            "identify the species from the image morphology",
            "identify the species from morphology",
            "from the image morphology using your knowledge",
            "using your knowledge of radiolarian taxonomy",
            "morphology-based guess",
            "identify from morphology",
        ]
        bad = [p for p in forbidden_phrases if p in prompt_lower]
        assert not bad, (
            f"audit 2026-09-04 llm-5: LLM-first prompt still tells the "
            f"model to NAME species from morphology when caption is "
            f"silent. Found forbidden phrases: {bad}. The model is not "
            f"a radiolarian expert — this rule causes fabricated "
            f"species names in DwC exports. Replace with explicit "
            f"\"set species: null\" rule."
        )

    def test_species_null_when_caption_silent(self):
        """The prompt must explicitly tell the model to set
        ``species: null`` when the caption does not mention a
        species for a panel."""
        from rlpe.pipeline import RadiolarianPipeline

        prompt = RadiolarianPipeline._LLM_FIRST_SYSTEM_PROMPT
        prompt_lower = prompt.lower()
        # Required: the prompt must tell the model what to do
        # when the caption doesn't provide a species. Either:
        # (a) "set species: null", (b) "set species to null",
        # (c) "set \"species\" to null", or some equivalent.
        null_rules = [
            "species: null",
            "species to null",
            "species field to null",
            '"species": null',
            "emit species as null",
            "set species as null",
        ]
        matched = any(rule in prompt_lower for rule in null_rules)
        assert matched, (
            f"audit 2026-09-04 llm-5: prompt must tell the model to "
            f"emit species:null when caption is silent. None of the "
            f"required phrases found. Searched: {null_rules}"
        )

    def test_no_never_invent_rule_contradiction(self):
        """The prompt already says 'NEVER invent species names' —
        the morphology-guess rule directly contradicts it. After
        the fix, the contradiction is gone (the morphology rule is
        either removed OR explicitly negated).

        The fix can be expressed as either:
          (a) Remove the morphology rule entirely, OR
          (b) Replace it with "Do NOT guess the species from image
              morphology — set species: null".

        Both forms are acceptable. The contract is: any rule
        that references morphology must explicitly NEGATE the
        act of naming-from-morphology (i.e. include "do not",
        "never", "must not", etc. before the morphology reference).
        """
        from rlpe.pipeline import RadiolarianPipeline

        prompt = RadiolarianPipeline._LLM_FIRST_SYSTEM_PROMPT
        prompt_lower = prompt.lower()

        has_never_invent = "never invent" in prompt_lower
        # Find every rule (bullet point) that mentions morphology.
        # Each rule must negate the act of naming from morphology.
        # Bullets start with "-" or are unindented statements.
        rules = re.split(r"\n\s*-\s*", "\n" + prompt)
        bad_rules = []
        for rule in rules:
            rule_lower = rule.lower()
            if "morphology" not in rule_lower:
                continue
            has_negation = any(
                neg in rule_lower
                for neg in ["do not", "don't", "never", "must not", "mustn't"]
            )
            if not has_negation:
                bad_rules.append(rule.strip()[:200])
        assert not (has_never_invent and bad_rules), (
            f"audit 2026-09-04 llm-5: prompt contains both "
            f"'NEVER invent species' AND a non-negated morphology "
            f"rule — these directly contradict. The morphology "
            f"rule must be either removed or negated. "
            f"Bad morphology rules: {bad_rules}. "
            f"Prompt excerpt: {prompt[:600]!r}"
        )