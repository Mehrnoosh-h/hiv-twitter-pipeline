"""Merge GPT rates, survey, and policy data into one integrated dataset."""

import argparse
from hiv_twitter_pipeline.data_processing import build_integrated_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpt_csv",      help="Path to GPT-rate CSV")
    parser.add_argument("survey_xlsx",  help="Path to HIV Vax Survey Excel workbook")
    parser.add_argument("policy_xlsx",  help="Path to Policy Coding Master Sheet Excel workbook")
    parser.add_argument("output_csv",   help="Path for the integrated output CSV")
    parser.add_argument("--policy-sheet", default="Master",
                        help="Sheet name in the policy workbook (default: Master)")
    args = parser.parse_args()

    df = build_integrated_dataset(
        gpt_csv_path=args.gpt_csv,
        survey_xlsx_path=args.survey_xlsx,
        policy_xlsx_path=args.policy_xlsx,
        policy_sheet=args.policy_sheet,
    )
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df):,} rows → {args.output_csv}")


if __name__ == "__main__":
    main()
