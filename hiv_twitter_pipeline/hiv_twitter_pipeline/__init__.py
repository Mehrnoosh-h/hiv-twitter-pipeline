"""
hiv_twitter_pipeline
====================
End-to-end pipeline for processing geo-tagged HIV-related tweets,
building GPT-rate variables, integrating survey/policy data, and
running fine-tuned model validation.

Sub-packages
------------
data_processing  – FIPS cleaning, GPT-rate variable creation, dataset integration
annotation       – OpenAI prompt templates and fine-tune configuration
validation       – Cross-validation, prompt building, and response collection
utils            – Shared helpers (FIPS, value cleaning, state mappings)
"""

from hiv_twitter_pipeline import data_processing, annotation, validation, utils

__all__ = ["data_processing", "annotation", "validation", "utils"]
