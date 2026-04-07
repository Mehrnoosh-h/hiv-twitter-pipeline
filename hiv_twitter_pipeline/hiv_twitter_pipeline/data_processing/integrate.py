"""
Integrated dataset assembly.

Merges three sources:
    1. GPT-rate variables (county / state aggregates)
    2. HIV Vax Survey (pre-screen + follow-up)
    3. Policy coding master sheet
"""

import numpy as np
import pandas as pd

from hiv_twitter_pipeline.utils.fips import pad_fips
from hiv_twitter_pipeline.utils.states import STATE_ABBREV_TO_NAME

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Columns retained from the GPT-rate data frame after deduplication.
GPT_KEEP_COLS: list[str] = [
    "state", "county_fips", "state_fips",
    "county_denominator", "state_denominator",
    "county_info_any", "county_inst3_any", "county_inst4_any",
    "county_negt_any", "county_random_hiv",
    "state_info_any", "state_inst3_any", "state_inst4_any",
    "state_negt_any", "state_random_hiv",
]

#: PII / duplicate columns dropped from the survey before merging.
SURVEY_DROP_COLS: list[str] = [
    "First Name", "Middle Name", "Last Name", "Email", "Address",
    "email", "phone", "firstname", "midname", "lastname", "email2",
    "zipcode", "secondphone", "address1", "address2", "address3",
    "firstname.1", "midname.1", "lastname.1", "email.1", "email2.1",
    "phone.1", "phone2", "dob.1", "age",
    "address_street", "address_state", "address_zip",
    "socialhandle", "secondname.1", "second_phone2",
    "secondemail2.1", "city",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_gpt_rates(csv_path: str) -> pd.DataFrame:
    """Load and deduplicate the GPT-rate variables CSV.

    Parameters
    ----------
    csv_path:
        Path to ``Twitter_GPTrates_2024.csv`` (or equivalent).
    """
    df = pd.read_csv(csv_path, dtype={"county_fips": str, "state_fips": str})
    available = [c for c in GPT_KEEP_COLS if c in df.columns]
    df = df[available].drop_duplicates(subset="county_fips", keep="first")
    return df


def load_survey(xlsx_path: str) -> pd.DataFrame:
    """Load and clean the HIV Vax Survey workbook.

    Merges the ``PreScreen_Survey0`` and ``Survey1`` sheets on ``Record ID``,
    removes PII columns, pads FIPS codes, and maps state abbreviations to
    full names.

    Parameters
    ----------
    xlsx_path:
        Path to ``HIV_Vax_Datasheet.xlsx`` (or equivalent).
    """
    survey0 = pd.read_excel(xlsx_path, sheet_name="PreScreen_Survey0")
    survey1 = pd.read_excel(xlsx_path, sheet_name="Survey1")
    df = survey0.merge(survey1, on="Record ID", how="outer")

    df.rename(columns={"fips_code": "county_fips"}, inplace=True)
    df["county_fips"] = pad_fips(df["county_fips"], 5)

    # Drop PII and duplicated columns
    existing = [c for c in SURVEY_DROP_COLS if c in df.columns]
    df = df.drop(columns=existing)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Map state abbreviations to full names
    if "state" in df.columns:
        df["state"] = df["state"].map(STATE_ABBREV_TO_NAME)

    return df


def load_policy(xlsx_path: str, sheet_name: str = "Master") -> pd.DataFrame:
    """Load the policy coding master sheet.

    Parameters
    ----------
    xlsx_path:
        Path to ``Policy_Coding_Master_Sheet_*.xlsx`` (or equivalent).
    sheet_name:
        Sheet to read (default ``'Master'``).
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df.rename(columns={"State": "state"}, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

def build_integrated_dataset(
    gpt_csv_path: str,
    survey_xlsx_path: str,
    policy_xlsx_path: str,
    policy_sheet: str = "Master",
) -> pd.DataFrame:
    """Merge survey, policy, and GPT-rate data into one integrated data frame.

    Merge order
    ~~~~~~~~~~~
    1. Survey  LEFT JOIN  Policy   on ``state``
    2. (result) LEFT JOIN  GPT     on ``county_fips``

    Parameters
    ----------
    gpt_csv_path:
        Path to the GPT-rate CSV produced by :func:`build_gpt_rate_df`.
    survey_xlsx_path:
        Path to the HIV Vax Survey Excel workbook.
    policy_xlsx_path:
        Path to the Policy Coding Master Sheet Excel workbook.
    policy_sheet:
        Sheet name within the policy workbook.

    Returns
    -------
    pd.DataFrame
        Fully integrated data frame ready for analysis.
    """
    gpt_df = load_gpt_rates(gpt_csv_path)
    survey_df = load_survey(survey_xlsx_path)
    policy_df = load_policy(policy_xlsx_path, sheet_name=policy_sheet)

    merged = survey_df.merge(policy_df, on="state", how="left")
    integrated = merged.merge(gpt_df, on="county_fips", how="left")

    # Resolve duplicate 'state' columns introduced by the double merge
    if "state_x" in integrated.columns:
        integrated = integrated.rename(columns={"state_x": "state"})
    if "state_y" in integrated.columns:
        integrated = integrated.drop(columns=["state_y"])

    return integrated
