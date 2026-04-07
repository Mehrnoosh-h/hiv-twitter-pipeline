"""
OpenAI prompt templates and fine-tune configuration.

Usage
-----
>>> from hiv_twitter_pipeline.annotation.prompts import build_prompt
>>> msgs = build_prompt(id_key="tgt", annotation_type="rec_act", q_key="act1",
...                     tweet_text="Get tested today — know your status!")
>>> print(msgs[0]["content"])
"""

import os

# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

#: Set via the ``OPENAI_API_KEY`` environment variable or replace directly.
API_KEY: str = os.environ.get("OPENAI_API_KEY", "")  # ← set your key here

# ---------------------------------------------------------------------------
# Identity / persona strings
# ---------------------------------------------------------------------------

#: System-role identity strings keyed by persona code.
ID_STRS: dict[str, str] = {
    "tgt": (
        "You are a member of a population that is a high priority target for efforts "
        "related to reducing incidents of the Human Immunodeficiency Virus "
        "(For ex: Men who have sex with men)."
    ),
}

# ---------------------------------------------------------------------------
# Question prompts
# ---------------------------------------------------------------------------

#: Individual rating questions keyed by short code.
Q_PROMPTS: dict[str, str] = {
    "act1": (
        "How likely is it that you would share this message with your "
        "social network if you encountered it on social media?"
    ),
    "act2": (
        "How likely do you think it is that this message was specifically "
        "designed for gay or bisexual men"
    ),
    "act3": "How strongly do you feel that this message is convincing?",
    "act4": (
        "How strongly do you feel this message gives the audience a clear indication "
        "of what to do about HIV prevention or testing"
    ),
    "act5": (
        "How strongly do you feel this message points to concrete resources that may "
        "help to implement HIV prevention or testing"
    ),
    "eff1": "How strongly do you agree that this message is truthful",
    "eff2": (
        "How strongly do you agree that this message seems to have been designed for "
        "the benefit of people with high risk of HIV (For ex: Men who have sex with men)"
    ),
    "eff3": (
        "How strongly do you agree that this message has the potential to change the "
        "behaviour of people with high risk of HIV (For ex: Men who have sex with men)"
    ),
    "eff4": (
        "How strongly do you agree that this message has the potential to change "
        "the behaviour of people with high risk of HIV (For ex: Men who have sex with men)"
    ),
}

# ---------------------------------------------------------------------------
# Annotation-type rating scale descriptions
# ---------------------------------------------------------------------------

#: Per-annotation-type instructions that describe the rating scale.
ANNOTATION_TYPE_PROMPTS: dict[str, str] = {
    "rec_act": (
        "You will rate these tweets with integer values ranging from 1 to 4, where "
        '1 is "Not at all" and 4 is "Definitely", based on the following question.'
    ),
    "avg_act": (
        "You will rate these tweets from 1 to 4, where "
        '1 is "Not at all" and 4 is "Definitely", based on the following question.'
    ),
    "rec_eff": (
        "You will rate these tweets with integer values ranging from 1 to 3, where "
        '1 is "Disagree", 2 is "Neither agree nor disagree", and 3 is "Agree", '
        "based on the following question."
    ),
    "avg_eff": (
        "You will rate these tweets from 1 to 3, where "
        '1 is "Disagree", 2 is "Neither agree nor disagree", and 3 is "Agree", '
        "based on the following question."
    ),
}

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(
    id_key: str,
    annotation_type: str,
    q_key: str,
    tweet_text: str,
) -> list[dict[str, str]]:
    """Build an OpenAI chat-completion message list.

    Parameters
    ----------
    id_key:
        Key into :data:`ID_STRS` (e.g. ``'tgt'``).
    annotation_type:
        Key into :data:`ANNOTATION_TYPE_PROMPTS` (e.g. ``'rec_act'``).
    q_key:
        Key into :data:`Q_PROMPTS` (e.g. ``'act1'``).
    tweet_text:
        The raw tweet content to rate.

    Returns
    -------
    list[dict[str, str]]
        A two-element list suitable for ``client.chat.completions.create(messages=...)``.

    Raises
    ------
    KeyError
        If any of the keys are not found in their respective dictionaries.
    """
    system_content = (
        f"{ID_STRS[id_key]} "
        f"{ANNOTATION_TYPE_PROMPTS[annotation_type]} "
        f"{Q_PROMPTS[q_key]}"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f'Rate the following tweet "{tweet_text}"'},
    ]
