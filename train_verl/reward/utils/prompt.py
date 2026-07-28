"""Judge prompt templates for judge-model scoring (translation + VQA)."""

JUDGE_SYSTEM_PROMPT = "You are a helpful and precise assistant for checking the consistency of answers."


TRANSLATION_JUDGE_PROMPT = """Given the **target language**, **source text**, **reference translation**, the scoring criteria and the 0-5 scoring rubric below, rate the <model translation> in **Model under Evaluation** by following the format shown in **Scoring Example**. Output ONLY the "Reasoning" and "Final Score" fields; do not output anything else.

[Scoring Criteria]
1. Semantic accuracy: the translation conveys the meaning of the source fully and correctly.
2. Fluency: the translation reads naturally in the target language.
3. Cultural adequacy: word choice is appropriate to the target-language conventions.
4. Terminology consistency: technical terms and well-known names of people / organizations / institutions are rendered correctly and consistently.

[Scoring Rubric (0-5)]
Score strictly. Semantic accuracy is the foundation; when it is weak, other dimensions cannot compensate for it (their scores must not exceed the semantic-accuracy score).
Use the **reference translation** as a 4-point anchor. Translations at the level of the reference get around 4 points; translations that clearly surpass the reference may get 4.0-5.0.
Numbers and LaTeX formulas should NOT be translated. The <model answer> may contain image descriptions, formatting changes, or repetition of the source; ignore those and evaluate only the translated part.

Reference cut-offs:
- 0: severe errors (large chunks missing, unintelligible output, or unsafe content).
- 1: harmless but low quality (mistranslated keywords, many missing key clauses, or unnatural flow).
- 2: basically usable with a mid-tier quality, individual non-technical terms or lesser-known names mistranslated, minor omissions of non-essential words.
- 3: consistently good across all dimensions, fluent and accurate, wording fits the context.
- 4.0-5.0: near-perfect across all dimensions, semantically accurate, fluent, idiomatic, and culturally appropriate.

[Scoring Example]
Reasoning: [[one concise sentence explaining the score]]
Final Score: 3.575

**User question**: {question}
**Target language**: {reference}
**Source text**:
{text}
**Reference translation**:
{gt}

[Model under Evaluation]
<model translation>:
{answer}
"""


VQA_JUDGE_PROMPT = """
# Task
You will see a question and two corresponding answers: [Standard Answer] and [Model Answer]. Decide whether the two answers agree in core meaning.

# Procedure
1. **Extract the core information** from [Standard Answer] and [Model Answer]: keep only the content that directly answers the question, ignoring pleasantries, prefixes, and summaries.
2. **Judge consistency**:
    * **Semantic consistency (default)**: the two answers convey the same core information. Differences in wording, order, or level of detail are allowed as long as they do not change the meaning, drop key information, or cause misunderstanding.
    * **Character consistency (special case)**: when the question asks for exact recitation (e.g. "What is the text in the image?"), the two answers must match exactly on characters and important punctuation.
3. **Output**:
    * **Reasoning**: one sentence explaining why you judged the two answers as consistent or inconsistent.
    * **Judgement**:
        * If consistent, output `Judgement: 1`
        * If inconsistent, output `Judgement: 0`

[Question]: {question}
[Standard Answer]: {gt}
[Model Answer]: {answer}
Reasoning:
"""


JUDGE_PROMPTS = {
    "translation": {
        "system_prompt": JUDGE_SYSTEM_PROMPT,
        "template": TRANSLATION_JUDGE_PROMPT,
    },
    "vqa": {
        "system_prompt": JUDGE_SYSTEM_PROMPT,
        "template": VQA_JUDGE_PROMPT,
    },
}
