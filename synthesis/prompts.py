# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prompt templates for multilingual instruction-following data synthesis.

Adapted from the IFDecorator flywheel (arXiv:2508.04632). Every template here has
a documented delta from the original in `modules/enhance/`; see
docs/SYNTHESIS_PIPELINE.md for the full derivation.

Two conventions apply throughout, and they are the main departure from upstream:

1. META-PROMPT IN ENGLISH, CONTENT IN THE TARGET LANGUAGE.
   The instructions to the generator model are English; the instruction being
   built, and the constraint text it emits, are in the target language. This
   decouples the generator's meta-task competence from its competence in the
   target language, and keeps the parse markers (`#constraint:`, `**Final
   Verification:**`) stable across all 24 languages. It is a recorded decision,
   not an accident - localizing the meta-prompts too is a legitimate alternative,
   but it changes what "difficulty" means and must be measured before adopting.

2. EVERY TEMPLATE IS EXPLICIT ABOUT LANGUAGE.
   Upstream had no notion of a target language, so nothing downstream could tell
   a French constraint from an English one. Here `language` is a required render
   argument wherever the output is content rather than a verdict.
"""

from __future__ import annotations

import random
from typing import Any


# Human-readable names for the languages multilingual_if supports. Used to fill
# {language} so the generator is told "Greek", not "el".
LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "cs": "Czech",
    "sk": "Slovak",
    "hu": "Hungarian",
    "ro": "Romanian",
    "fi": "Finnish",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "sv": "Swedish",
    "da": "Danish",
    "bg": "Bulgarian",
    "uk": "Ukrainian",
    "ru": "Russian",
    "el": "Greek",
    "ca": "Catalan",
    "mt": "Maltese",
}


def language_name(code: str) -> str:
    try:
        return LANGUAGE_NAMES[code]
    except KeyError:
        raise KeyError(
            f"{code!r} has no language name. multilingual_if supports "
            f"{sorted(LANGUAGE_NAMES)}; anything else is unverifiable and must not be synthesized."
        ) from None


# ---------------------------------------------------------------------------
# W0 - quality gate
# ---------------------------------------------------------------------------
# Upstream: modules/enhance/w0.2_quality_filter.py::quality_en_prompt
# Delta: adds the language check. Upstream ran a separate langdetect == "en"
#        filter before this; here the gate and the language check are one call,
#        and a prompt in the wrong language is a quality failure like any other.

QUALITY_GATE = """Assess whether the instruction below is sufficiently clear and actionable, and whether it is written in {language}.

Respond YES if it can be reasonably understood and executed without major issues AND it is written in {language}.
Respond NO if it has critical flaws such as:
- Complete lack of clarity in purpose
- Contradictory requirements
- Unintelligible language
- Written in a language other than {language}

###[Instruction]
<Instruction>
{prompt}
</Instruction>

###[Evaluation Requirements]
1. Detailed analysis
2. Conclude with final verdict using strict formatting:
**Final Verification:** <YES/NO>
"""


# ---------------------------------------------------------------------------
# W1 - decompose
# ---------------------------------------------------------------------------
# Upstream: modules/enhance/w1_decompose.py::decompose_en_prompt
# Deltas:
#   * few-shot examples are rendered in the target language, so the model is not
#     primed to answer in English about a Greek instruction
#   * output markers stay English and fixed, so one parser serves all languages
#   * explicitly forbids translating the extracted text - upstream never had to
#     say this because everything was English, and models will "helpfully"
#     translate constraints into English otherwise, which silently destroys the
#     link between the constraint and the instruction it came from

DECOMPOSE = """You are a prompt engineering specialist. Given a prompt written in {language}, perform the following clearly defined tasks:

### Tasks:
1. **Extract Task Description**: Clearly state the primary objective of the prompt.
2. **List Constraints**: Identify and list explicit rules, formats, styles, conditions, or limitations specified in the prompt. If none exist, output `NULL`.
3. **Determine Input Requirements**: Identify any specific data or inputs explicitly required from the user. If none exist, output `NULL`.

### Processing Guidelines:
- Use `NULL` for Constraints and Input fields if the prompt does not explicitly mention them.
- Do not duplicate content between Task Description, Constraints, and Input fields.
- Ensure extracted information is semantically consistent with the original prompt.
- **Write the extracted content in {language}, the same language as the prompt. Do NOT translate it.**
- Keep the section markers (`#Task Description:`, `#Constraints:`, `#Input:`) in English exactly as shown.

### Output Format:
#Task Description: [Concise statement of the primary objective, in {language}]
#Constraints: [List constraints clearly, in {language}] or NULL
#Input: [Specific user-provided data required, in {language}] or NULL

### Example
<Beginning of Example>
---INPUT---
#Prompt: {example_prompt}
---OUTPUT---
#Task Description: {example_task}
#Constraints:
{example_constraints}
#Input: {example_input}
<End of Example>

Now, analyze and respond to the provided prompt following the same method:

---INPUT---
#Prompt: {prompt}
---OUTPUT---
"""


# ---------------------------------------------------------------------------
# W2 - classify constraints
# ---------------------------------------------------------------------------
# Upstream: modules/enhance/w2_classify_constraints.py::classification_en_prompt
# Deltas:
#   * "hard/soft" renamed to "programmatic/rubric" - the upstream naming is
#     actively confusing downstream, where `hard_constraints_checklist` actually
#     held checklists for the LLM-JUDGED constraints, not the programmatic ones
#   * the definition of "programmatic" is now grounded in a concrete capability
#     list rather than left to the model's imagination, because a constraint
#     classified programmatic that no checker implements is dead weight
#   * classifies the whole constraint list in ONE call rather than one call per
#     constraint (upstream: w2_classify_constraints.py:108 loops per constraint)

CLASSIFY_CONSTRAINTS = """You are a prompt engineering specialist. Classify each constraint below as **programmatic** or **rubric**.

### Definitions:
1. **programmatic** - can be checked by a deterministic program with no judgement, using only the response text. The available programmatic checks are:
   - word / sentence / paragraph counts
   - presence, absence or frequency of a specific word
   - letter frequency and letter counts
   - all-lowercase, all-uppercase, count of ALL-CAPS words
   - required first or last word of the response or of a sentence
   - bullet lists, highlighted sections, numbered sections, titles wrapped in <<>>
   - JSON output, placeholders in [square brackets], wrapping the answer in quotes
   - a required postscript marker, a required closing phrase
   - response written entirely in a named language
   - repeating or copying a given span of text
   - no commas / no periods / no exclamation marks

2. **rubric** - requires judgement: tone, persona, style, creativity, coverage,
   factual adequacy, or anything needing interpretation of meaning.

### Rules:
- If a constraint would need a human or a language model to decide, it is **rubric**.
- If it is measurable by one of the listed checks, it is **programmatic**.
- When genuinely ambiguous, choose **rubric**. A constraint wrongly marked
  programmatic becomes an unenforceable reward signal; one wrongly marked rubric
  merely costs a judge call.

### Output Format:
One line per constraint, in the same order as the input, no extra commentary:
#1: <programmatic|rubric>
#2: <programmatic|rubric>
...

---Input---
#Prompt ({language}):
{prompt}

#Constraints:
{numbered_constraints}
---Output---
"""


# ---------------------------------------------------------------------------
# W3 - rubric generation (REPLACES the upstream checklist step)
# ---------------------------------------------------------------------------
# Upstream: modules/enhance/w3_add_checklist.py::add_checklist_en_prompt
# Status: REFERENCE ONLY. You have your own rubric generator and judge; this is
#         here so the pipeline is runnable end-to-end without it, and so the
#         contract the rest of the pipeline expects is written down somewhere.
# Deltas:
#   * emits a structured rubric with explicit pass criteria rather than a loose
#     markdown checklist, so a judge can be asked for a per-criterion verdict
#   * generated only for `rubric`-classified constraints. Upstream generated
#     checklists only for `hard`-typed ones and then zipped them against the FULL
#     constraint list, so any soft constraint appearing earlier shifted every
#     checklist onto the wrong constraint (check_instruction.py:136, cif.py:226).
#     Carrying the constraint text inside the rubric object makes that class of
#     bug impossible - never re-introduce a positional zip here.

RUBRIC_GENERATION = """Design an evaluation rubric that determines whether one specific constraint is satisfied by a response.

FOCUS SOLELY on the target constraint. Ignore every other requirement in the instruction.

### Rules:
- Produce 2 to 5 criteria. Each must be independently checkable by reading the response.
- Each criterion must be a yes/no question with an unambiguous answer.
- Do not restate the constraint as a criterion; decompose it.
- The response under evaluation will be in {language}. Write the criteria in English, but they must be applicable to a {language} response - do not assume English word order, script, or punctuation.

**[Instruction]**
<Instruction>
{instruction}
</Instruction>

**[Target Constraint]**
<TargetConstraint>
{constraint}
</TargetConstraint>

### Output Format:
#criterion_1: <yes/no question>
#criterion_2: <yes/no question>
...
#pass_rule: <ALL|MAJORITY>
"""


# ---------------------------------------------------------------------------
# W4 - instruction evolution
# ---------------------------------------------------------------------------
# Upstream: modules/enhance/w4_evol.py::evol_dynamic_instruction_enchancement_en_prompt
#           (assembled at call time by parse_prompt_template)
# Deltas:
#   * the added constraint must be written in the target language and integrated
#     into a target-language instruction
#   * "measurable boundaries" guidance removed from the special rules: it pushed
#     the model toward pseudo-programmatic constraints that no checker
#     implements. Programmatic constraints are added separately and deliberately
#     from the verified taxonomy, not invented here.
#   * the randomization (shuffling categories, weighting by inverse frequency) is
#     kept - it is what keeps the constraint distribution from collapsing - and
#     is implemented in `render_evolution` below.

_EVOLUTION_FRAMEWORK = {
    "content_constraints": {
        "types": ["Open-scope", "Language", "Structural"],
        "examples": [
            "Add related subtask/question",
            "Specify language complexity level",
            "Require specific format/structure",
        ],
    },
    "situation_constraints": {
        "types": ["Role-based", "Scenario-specific", "Story-driven"],
        "examples": [
            "Define role/persona requirements",
            "Set environmental/contextual parameters",
            "Add plot/character development elements",
        ],
    },
    "style_constraints": {
        "types": ["Tonal", "Structural", "Creative"],
        "examples": [
            "Specify emotional tone",
            "Request specific narrative style",
            "Add ambiguity/humor elements",
        ],
    },
}

_EVOLUTION_GUIDELINES = [
    "Preserve all non-text elements (tables, code, etc.) from the original",
    "Maintain logical coherence and human readability",
    "Add only 10-20 meaningful words for constraint integration",
    "Select constraints based on instruction type and enhancement potential",
]

_EVOLUTION_SPECIAL_RULES = [
    "Maintain original instruction intent while adding specificity",
    "Avoid overlapping/conflicting constraints in a single enhancement",
    "The constraint must require judgement to evaluate, not a word count or a format rule",
]

_EVOLUTION_EXAMPLE_TYPES = [
    "Situation, Scenario-specific",
    "Situation, Story-driven",
    "Situation, Role-based",
    "Style, Tonal",
    "Style, Structural",
    "Style, Creative",
    "Content, Structural",
    "Content, Language",
    "Content, Open-scope",
]

_EVOLUTION_HEADER = """You are an Instruction Enhancement Expert. Analyze the **Original Instruction**, which is written in {language}, and select the most appropriate enhancement category from [Content, Situation, Style]. Apply ONE relevant constraint to refine the instruction while following these guidelines:"""


def render_evolution(
    instruction: str,
    language: str,
    distribution: dict[str, dict[str, int]] | None = None,
    rng: random.Random | None = None,
) -> str:
    """Assemble the evolution prompt, re-randomized per call.

    The randomization is deliberate and load-bearing. A fixed template collapses
    the constraint distribution onto whichever category the model prefers;
    shuffling the presentation order, and optionally weighting by inverse
    observed frequency, keeps it spread. `distribution` maps
    category -> {type: count_so_far}; pass the running counts from previous
    rounds to actively correct an imbalance.
    """
    rng = rng or random.Random()
    name = language_name(language)
    sections = [_EVOLUTION_HEADER.format(language=name)]

    sections.append("\n### Guidelines:")
    sections.extend(f"- {g}" for g in _EVOLUTION_GUIDELINES)

    sections.append("\n### Enhancement Framework:")
    for category, details in rng.sample(
        list(_EVOLUTION_FRAMEWORK.items()), len(_EVOLUTION_FRAMEWORK)
    ):
        types = list(details["types"])
        examples = list(details["examples"])
        indices = list(range(len(types)))
        if distribution and category in distribution:
            # Weighted ordering WITHOUT replacement. Upstream used
            # random.choices here (w4_evol.py:147), which samples with
            # replacement - so an over-represented type could be duplicated and
            # an under-represented one dropped from the prompt entirely, meaning
            # the model could never pick the very type the weighting was trying
            # to promote. Every type must survive; only the order is biased.
            weights = {
                i: 1.0 / (distribution[category].get(types[i], 0) + 1) for i in indices
            }
            ordered = []
            remaining = dict(weights)
            while remaining:
                pick = rng.choices(
                    list(remaining), weights=list(remaining.values()), k=1
                )[0]
                ordered.append(pick)
                del remaining[pick]
            indices = ordered
        else:
            rng.shuffle(indices)
        sections.append(f"\n{category.replace('_', ' ').title()}:")
        sections.append(f"Types: {', '.join(types[i] for i in indices)}")
        sections.append("Examples:")
        sections.extend(f"- {examples[i]}" for i in indices)

    rules = list(_EVOLUTION_SPECIAL_RULES)
    rng.shuffle(rules)
    sections.append("\n### Special Rules:")
    sections.extend(f"- {r}" for r in rules)

    sections.append("\n### Output Requirements:")
    sections.append(
        "#rationale: Brief explanation of constraint selection (20 words, in English)"
    )
    sections.append(
        f"#constraint_type: Format as 'Category, Type' (e.g. '{rng.choice(_EVOLUTION_EXAMPLE_TYPES)}')"
    )
    sections.append(f"#constraint: The constraint to add, written in {name}")
    sections.append(
        f"#enhanced_instruction: The full modified instruction, written in {name}"
    )

    sections.append(
        f"\n---INPUT---\n#**Original Instruction** ({name}):\n<Instruction>\n{instruction}\n</Instruction>\n\n---OUTPUT---\n"
    )
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# W4 - keyword extraction (for programmatic keyword constraints)
# ---------------------------------------------------------------------------
# Upstream: modules/enhance/w4_evol.py::evol_get_keywords_en_prompt
# Deltas:
#   * keywords must be in the target language and must be SURFACE FORMS THAT
#     ALREADY APPEAR in the seed response. This is not cosmetic: keyword matching
#     is surface-form (stemming was deliberately not implemented), so a keyword
#     the model would have to inflect will never be matched. Upstream fell back
#     to instructions_util.WORD_LIST, an English word list, which is unusable here.
#   * asks for single tokens - upstream then did keywords.split(" ")[0] anyway,
#     silently truncating multi-word keywords

EXTRACT_KEYWORDS = """You are given an instruction and a reference answer, both written in {language}. Propose keywords that could plausibly be required to appear in an answer to the instruction.

### Rules:
1. Every keyword MUST appear verbatim in the reference answer, in exactly the form you output.
2. Output single words only - no phrases, no punctuation.
3. Output the words in {language}, in the exact surface form found in the reference answer. Do NOT translate them and do NOT give a dictionary/base form.
4. Prefer content words (nouns, verbs, adjectives) over function words.

### Output Format
**Thinking Process:** <brief reasoning, in English>
**Keywords:** <a Python list of strings, e.g. ['word1', 'word2', 'word3']>

---INPUT---
**Instruction:**
<Instruction>
{instruction}
</Instruction>

**Reference Answer:**
<Answer>
{response}
</Answer>
---OUTPUT---
"""


# ---------------------------------------------------------------------------
# W4 - containment check
# ---------------------------------------------------------------------------
# Upstream: modules/enhance/w4_evol.py::judge_include_en_prompt
# Status: KEEP AS FALLBACK ONLY.
# Delta / warning: upstream spent one LLM call per merge to answer "is text2
#   inside text1", twice per evolution attempt. That is the single largest
#   avoidable cost in the flywheel. Use synthesis.matching.contains() first -
#   NFC + whitespace-normalized substring, then a similarity ratio - and fall
#   back to this prompt only when the ratio lands in the ambiguous band.

CONTAINMENT_CHECK = """### Task: Text Matching Verification
You are given two pieces of text in {language}: **Text 1** and **Text 2**. Determine whether **Text 2** appears within **Text 1**, allowing for minor rewording.

### Output Instructions:
1. If Text 2 is largely present within Text 1, allowing for some minor differences, output YES.
2. Otherwise, output NO.

### Output Format
Do not provide any additional explanation. Output your final verdict using strict formatting:
**Final Verification:** <YES/NO>
---INPUT---
**Text 1:**
<text1>
{text1}
</text1>

**Text 2:**
<text2>
{text2}
</text2>
---OUTPUT---
"""


# ---------------------------------------------------------------------------
# W5 - judging (difficulty tagging, and the RL-time rubric reward)
# ---------------------------------------------------------------------------
# Upstream: modules/enhance/check_instruction.py::judge_overall_en_prompt
# Deltas:
#   * the "Safety & Compliance Check" step is removed. It conflated safety with
#     instruction-following, so a refusal and a formatting miss produced the same
#     verdict, and difficulty became partly a safety signal. Filter safety
#     separately if you need it.
#   * explicit statement that the response is in {language} and that being in
#     that language is NOT what is being judged here - response_language is a
#     programmatic check, and judging it twice double-counts it

JUDGE_INTENT = """### Structured Evaluation Protocol
Perform a rigorous analysis of the instruction-response pair through these sequential checks. Both are written in {language}.

1. Instruction-Response Alignment
- Verify explicit understanding of core instruction objectives
- Check for missing required components from the instruction
2. Logical Coherence Evaluation
- Trace logical flow from instruction premises to response conclusions
- Detect reasoning gaps or unwarranted assumptions
- Flag contradictions within the response
3. Context-aware Instruction Verification
- Analyze response against instruction type:
  - Query-type: Verify question resolution completeness
  - Task-type: Validate step-by-step executable logic
  - Creative-type: Assess objective-aligned originality

### Evaluation Parameters
- Strict true/false determination for each checkpoint
- Zero tolerance for partial fulfillment
- Mandatory failure for any single unmet criterion
- Do NOT evaluate which language the response is written in; that is checked separately.
- Do NOT evaluate formatting, length or word-choice rules; those are checked separately.

### Evaluation Target
**Instruction:**
<Instruction>
{instruction}
</Instruction>

**Response:**
<Response>
{response}
</Response>

### Output Format
First, present analysis in ordered checklist format. Then conclude with the final verdict using strict formatting, in English:
**Final Verification:** <YES/NO>
"""

# Upstream: modules/enhance/check_instruction.py::judge_checklist_en_prompt
# Status: REFERENCE ONLY - your judge replaces this.
# Delta: takes a structured rubric and returns a per-criterion verdict, so a
#        partial credit signal is available. Upstream returned one boolean for
#        the whole checklist, which discards most of the information.

JUDGE_RUBRIC = """You are an impartial judge. Evaluate whether the *target constraint* is satisfied by the *response*, using the *rubric*. Focus solely on the target constraint and disregard every other requirement in the instruction.

The instruction and response are written in {language}.

### Instruction:
<Instruction>
{instruction}
</Instruction>

### Target Constraint:
<TargetConstraint>
{constraint}
</TargetConstraint>

### Response:
<Response>
{response}
</Response>

### Rubric:
<Rubric>
{rubric}
</Rubric>

### Output Format:
Give a verdict for each criterion, then the overall verdict. Use strict formatting, in English:
#criterion_1: <YES/NO>
#criterion_2: <YES/NO>
...
**Final Verification:** <YES/NO>
"""


# ---------------------------------------------------------------------------
# Postprocess - domain filter
# ---------------------------------------------------------------------------
# Upstream: modules/postprocess/post_filter.py::filter_en_prompt
# Deltas:
#   * upstream generated a fresh response before classifying, doubling the cost;
#     classify from the instruction alone
#   * upstream's template embeds a literal JSON example containing braces and was
#     applied with str.replace rather than str.format. Kept compatible here by
#     using named placeholders that do not collide with the JSON braces, so it is
#     safe with .format() as long as you escape the example - which this does.

DOMAIN_FILTER = """You are a professional data labeling expert. Classify the instruction below into exactly one category.

### Categories:
1. **Math Problem** - asks to solve a math problem, perform calculations, or apply mathematical reasoning.
2. **Code Task** - relates to programming: writing, reviewing, explaining or debugging code.
3. **Reasoning Task** - a logic puzzle, brain teaser, or task whose value is the reasoning chain.
4. **Other** - anything else.

Instruction-following training data should be **Other**. The first three categories carry a correctness signal that competes with the formatting signal being trained, so they are filtered out.

### Input:
---Instruction ({language})---
<instruction>
{instruction}
</instruction>

### Output Format:
Reply with exactly one of: Math Problem, Code Task, Reasoning Task, Other
Then, on a second line, a one-sentence reason in English.
"""


def render(template: str, language: str, **kwargs: Any) -> str:
    """Fill a template, resolving `language` to its human-readable name.

    Raises rather than silently defaulting: an unknown language means the row
    should never have been synthesized, because nothing can verify it.
    """
    return template.format(language=language_name(language), **kwargs)
