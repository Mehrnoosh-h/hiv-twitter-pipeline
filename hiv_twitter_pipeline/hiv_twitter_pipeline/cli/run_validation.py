"""Run fine-tuned GPT validation for a given CV fold."""

import argparse
from hiv_twitter_pipeline.validation import ValidationConfig, run_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-params",   required=True,
                        help="Path to cv_recode_validation_params.json")
    parser.add_argument("--cv-param",      required=True, choices=["cv1","cv2","cv3","cv4","cv5"])
    parser.add_argument("--neg-fpath",     required=True, help="Path to neg_hiv_msgs.csv")
    parser.add_argument("--control-fpath", required=True, help="Path to control_msgs.txt")
    parser.add_argument("--op-rootpath",   required=True, help="Output directory")
    parser.add_argument("--val-rootpath",  default="",
                        help="Root directory prepended to the validation CSV filename")
    parser.add_argument("--annotation-type", default="rec_rating")
    args = parser.parse_args()

    cfg = ValidationConfig.from_json(
        json_path=args.json_params,
        cv_param=args.cv_param,
        neg_fpath=args.neg_fpath,
        control_fpath=args.control_fpath,
        op_rootpath=args.op_rootpath,
        validation_rootpath=args.val_rootpath,
        annotation_type=args.annotation_type,
    )
    run_validation(cfg)


if __name__ == "__main__":
    main()
