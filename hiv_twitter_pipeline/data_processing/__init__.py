"""
data_processing
---------------
GPT-rate variable creation and integrated dataset assembly.
"""

from hiv_twitter_pipeline.data_processing.gpt_rate_vars import (
    GPT_ANNOTATION_COLS,
    gen_any_cols,
    compute_county_rates,
    compute_state_rates,
    build_gpt_rate_df,
)
from hiv_twitter_pipeline.data_processing.integrate import build_integrated_dataset

__all__ = [
    "GPT_ANNOTATION_COLS",
    "gen_any_cols",
    "compute_county_rates",
    "compute_state_rates",
    "build_gpt_rate_df",
    "build_integrated_dataset",
]
