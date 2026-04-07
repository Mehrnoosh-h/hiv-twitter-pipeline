"""
annotation
----------
OpenAI prompt templates and fine-tune configuration for the HIV tweet
annotation pipeline.
"""

from hiv_twitter_pipeline.annotation.prompts import (
    API_KEY,
    ID_STRS,
    Q_PROMPTS,
    ANNOTATION_TYPE_PROMPTS,
    build_prompt,
)

__all__ = [
    "API_KEY",
    "ID_STRS",
    "Q_PROMPTS",
    "ANNOTATION_TYPE_PROMPTS",
    "build_prompt",
]
