"""Build GPT-rate variables from an annotated tweets CSV."""

import argparse
from hiv_twitter_pipeline.data_processing import build_gpt_rate_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", help="Path to annotated tweets CSV")
    parser.add_argument("output_csv", help="Path for the enriched output CSV")
    args = parser.parse_args()

    df = build_gpt_rate_df(args.input_csv)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df):,} rows → {args.output_csv}")


if __name__ == "__main__":
    main()
