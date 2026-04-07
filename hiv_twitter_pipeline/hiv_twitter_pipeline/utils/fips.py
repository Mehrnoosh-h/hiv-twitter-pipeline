"""FIPS code helpers."""

import numpy as np
import pandas as pd
import addfips

_af = addfips.AddFIPS()


def add_fips_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a ``state_fips`` column derived from the ``state`` column."""
    df = df.copy()
    df["state_fips"] = df["state"].apply(
        lambda x: _af.get_state_fips(x.strip()) if pd.notnull(x) else None
    )
    return df


def pad_fips(series: pd.Series, width: int) -> pd.Series:
    """Convert a FIPS series to zero-padded strings of *width* digits.

    Non-numeric values become ``np.nan``.
    """
    return (
        pd.to_numeric(series, errors="coerce")
        .apply(lambda x: str(int(x)).zfill(width) if pd.notnull(x) else np.nan)
    )
