"""
GPT-rate variable creation.

This module turns a raw annotated-tweets CSV into county- and state-level
proportion variables for five semantic tweet categories:

    info_any      – any information about STI testing / prevention / care
    inst3_any     – any instructions (testing, prevention, care)
    inst4_any     – any instructions including false-prevention claims
    negt_any      – any negative public-health / testing content
    random_hiv    – HIV-related but lacking informational / instructional content
"""

import numpy as np
import pandas as pd

from hiv_twitter_pipeline.utils.cleaning import clean_values
from hiv_twitter_pipeline.utils.fips import add_fips_codes, pad_fips

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Raw annotation columns produced by the ChatGPT classification prompts.
GPT_ANNOTATION_COLS: list[str] = [
    "q_info_sti_tst",
    "q_info_sti_prev",
    "q_info_sti_care",
    "q_inst_sti_tst",
    "q_inst_sti_prev",
    "q_inst_sti_falseprev",
    "q_inst_sti_care",
    "q_negt_sti_tst",
    "q_negt_pub_hlt",
    "q_if_sti",
]

#: Derived binary-flag columns and the output prefix used when aggregating.
DERIVED_COLS: list[str] = [
    "info_any",
    "inst3_any",
    "inst4_any",
    "negt_any",
    "random_hiv",
]

#: Subset of tweet-level columns kept in the final enriched data frame.
TWEET_KEEP_COLS: list[str] = [
    "id_str", "text", "location_raw", "state", "county_fips", "created_at",
    "quote_count", "reply_count", "retweet_count", "favorite_count",
    "screen_name", "description", "professions", "followers_count",
    "friends_count",
] + DERIVED_COLS


# ---------------------------------------------------------------------------
# Core transformations
# ---------------------------------------------------------------------------

def gen_any_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add five derived binary columns to *df* and return the result.

    Columns added
    ~~~~~~~~~~~~~
    ``info_any``
        'Yes' when any of the three ``q_info_*`` prompts returned 'Yes'.
    ``inst3_any``
        'Yes' when any of ``q_inst_sti_tst``, ``q_inst_sti_prev``, or
        ``q_inst_sti_care`` returned 'Yes'.
    ``inst4_any``
        'Yes' when any of the four ``q_inst_*`` prompts returned 'Yes'
        (including ``q_inst_sti_falseprev``).
    ``negt_any``
        'Yes' when either ``q_negt_pub_hlt`` or ``q_negt_sti_tst`` returned
        'Yes'.
    ``random_hiv``
        'Yes' when the tweet is HIV-related (``q_if_sti == 'Yes'``) but
        contains neither informational nor instructional content.
    """
    df = df.copy()

    df["info_any"] = "No"
    df.loc[
        (df["q_info_sti_tst"] == "Yes")
        | (df["q_info_sti_prev"] == "Yes")
        | (df["q_info_sti_care"] == "Yes"),
        "info_any",
    ] = "Yes"

    df["inst3_any"] = "No"
    df.loc[
        (df["q_inst_sti_tst"] == "Yes")
        | (df["q_inst_sti_prev"] == "Yes")
        | (df["q_inst_sti_care"] == "Yes"),
        "inst3_any",
    ] = "Yes"

    df["inst4_any"] = "No"
    df.loc[
        (df["q_inst_sti_tst"] == "Yes")
        | (df["q_inst_sti_prev"] == "Yes")
        | (df["q_inst_sti_care"] == "Yes")
        | (df["q_inst_sti_falseprev"] == "Yes"),
        "inst4_any",
    ] = "Yes"

    df["negt_any"] = "No"
    df.loc[
        (df["q_negt_pub_hlt"] == "Yes") | (df["q_negt_sti_tst"] == "Yes"),
        "negt_any",
    ] = "Yes"

    df["random_hiv"] = "No"
    df.loc[
        (df["q_if_sti"] == "Yes")
        & (df["info_any"] == "No")
        & (df["inst4_any"] == "No"),
        "random_hiv",
    ] = "Yes"

    return df


def _binarise(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a copy of *df* with *cols* encoded as 1 (Yes) / 0 (other)."""
    df = df.copy()
    df[cols] = df[cols].applymap(lambda x: 1 if x == "Yes" else 0)
    return df


def compute_county_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute county-level proportions for each derived column.

    Parameters
    ----------
    df:
        Tweet-level data frame with ``county_fips``, ``county_denominator``,
        and the five DERIVED_COLS already present.

    Returns
    -------
    pd.DataFrame
        One row per county with columns
        ``county_fips`` and ``county_{col}`` for each col in DERIVED_COLS.
    """
    df_bin = _binarise(df, DERIVED_COLS)

    grouped = df_bin.groupby("county_fips")[DERIVED_COLS].sum().reset_index()
    grouped["county_denominator"] = (
        df.groupby("county_fips")["county_denominator"].first().values
    )

    for col in DERIVED_COLS:
        grouped[f"county_{col}"] = grouped[col] / grouped["county_denominator"]

    return grouped[["county_fips"] + [f"county_{col}" for col in DERIVED_COLS]]


def compute_state_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute state-level proportions for each derived column.

    Parameters
    ----------
    df:
        Tweet-level data frame with ``state``, ``state_denominator``, and
        the five DERIVED_COLS already present.

    Returns
    -------
    pd.DataFrame
        One row per state with columns
        ``state`` and ``state_{col}`` for each col in DERIVED_COLS.
    """
    df_bin = _binarise(df, DERIVED_COLS)

    grouped = df_bin.groupby("state")[DERIVED_COLS].sum().reset_index()
    grouped["state_denominator"] = (
        df.groupby("state")["state_denominator"].first().values
    )

    for col in DERIVED_COLS:
        grouped[f"state_{col}"] = grouped[col] / grouped["state_denominator"]

    return grouped[["state"] + [f"state_{col}" for col in DERIVED_COLS]]


def build_gpt_rate_df(raw_csv_path: str) -> pd.DataFrame:
    """Full pipeline: load → clean → derive flags → compute rates → merge.

    Parameters
    ----------
    raw_csv_path:
        Path to the annotated tweets CSV
        (``twitter2024_annotated.csv`` or equivalent).

    Returns
    -------
    pd.DataFrame
        Tweet-level data frame enriched with county- and state-level rate
        columns.  This is the data frame saved as ``GPT_twitter2024.csv``.
    """
    df = pd.read_csv(raw_csv_path, dtype={"id_str": str})

    # Clean annotation columns
    df[GPT_ANNOTATION_COLS] = df[GPT_ANNOTATION_COLS].applymap(clean_values)

    # Derive binary flags
    df = gen_any_cols(df)

    # Trim to relevant columns
    keep = [c for c in TWEET_KEEP_COLS if c in df.columns]
    df = df[keep]

    # FIPS codes
    df = add_fips_codes(df)
    df["state_fips"] = pad_fips(df["state_fips"], 2)
    df["county_fips"] = pad_fips(df["county_fips"], 5)

    # Normalise state names
    df["state"] = df["state"].str.title()
    df.loc[df["state"] == "District Of Columbia", "state"] = "District of Columbia"
    df = df.drop(columns=["code"], errors="ignore")

    # Denominators
    df["county_denominator"] = df["county_fips"].map(df["county_fips"].value_counts())
    df["state_denominator"] = df["state"].map(df["state"].value_counts())

    # Aggregate rates
    county_rates = compute_county_rates(df)
    state_rates = compute_state_rates(df)

    # Merge back onto tweet-level data
    df = df.merge(county_rates, on="county_fips", how="left")
    df = df.merge(state_rates, on="state", how="left")
    df = df.rename(columns={"created_at": "tweet_date"})

    return df
