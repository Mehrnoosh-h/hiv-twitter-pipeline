"""
Validation runner.

Orchestrates cross-validation across three fine-tuned GPT personas
(health, target-group, quantitative) against:
    * Validation tweets  (human-rated, split into CV folds)
    * Negative tweets    (HIV-unrelated control set)
    * Control messages   (synthetic neutral messages)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from os import path
from typing import Any

import openai
import pandas as pd

from hiv_twitter_pipeline.annotation.prompts import (
    API_KEY,
    ID_STRS,
    ANNOTATION_TYPE_PROMPTS,
    Q_PROMPTS,
    build_prompt,
)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationConfig:
    """All paths and model IDs needed for one CV fold.

    Attributes
    ----------
    cv_param:
        Fold identifier, one of ``'cv1'`` … ``'cv5'``.
    validation_fpath:
        Path to the fold-specific validation CSV.
    health_gpt_id:
        Fine-tuned model ID for the health-professional persona.
    quant_gpt_id:
        Fine-tuned model ID for the quantitative-rater persona.
    tgt_gpt_id:
        Fine-tuned model ID for the target-group persona.
    neg_fpath:
        Path to the negative HIV messages CSV.
    control_fpath:
        Path to the control messages text file.
    op_rootpath:
        Directory where output CSVs are saved.
    annotation_type:
        Rating type used for this run (e.g. ``'rec_rating'``).
    qs_to_validate:
        List of question codes to evaluate (e.g. ``['act_2', 'eff_1']``).
    """

    cv_param: str
    validation_fpath: str
    health_gpt_id: str
    quant_gpt_id: str
    tgt_gpt_id: str
    neg_fpath: str
    control_fpath: str
    op_rootpath: str
    annotation_type: str = "rec_rating"
    qs_to_validate: list[str] = field(default_factory=lambda: [
        "act_2", "act_3", "act_4", "act_5",
        "eff_1", "eff_2", "eff_3", "eff_4",
    ])

    @classmethod
    def from_json(
        cls,
        json_path: str,
        cv_param: str,
        neg_fpath: str,
        control_fpath: str,
        op_rootpath: str,
        validation_rootpath: str = "",
        **kwargs: Any,
    ) -> "ValidationConfig":
        """Load a :class:`ValidationConfig` from the JSON params file.

        Parameters
        ----------
        json_path:
            Path to ``cv_recode_validation_params.json``.
        cv_param:
            Fold key (``'cv1'`` … ``'cv5'``).
        neg_fpath, control_fpath, op_rootpath:
            Fixed paths passed through to the config.
        validation_rootpath:
            Root directory prepended to ``valid_fname`` from the JSON.
        **kwargs:
            Additional keyword arguments forwarded to the constructor.
        """
        with open(json_path) as f:
            params = json.load(f)[cv_param]

        validation_fpath = path.join(validation_rootpath, params["valid_fname"])
        return cls(
            cv_param=cv_param,
            validation_fpath=validation_fpath,
            health_gpt_id=params["health_gpt_id"],
            quant_gpt_id=params["quant_gpt_id"],
            tgt_gpt_id=params["tgt_gpt_id"],
            neg_fpath=neg_fpath,
            control_fpath=control_fpath,
            op_rootpath=op_rootpath,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_validation_data(
    cfg: ValidationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load and reshape all three input sources.

    Returns
    -------
    tuple of (validation_df, neg_df, control_msgs)
        ``validation_df`` – melted, with columns
            ``Content``, ``variable``, ``value``, ``qcode``,
            ``annotation_type``.
        ``neg_df`` – negative tweets long-form with the same extra columns.
        ``control_msgs`` – list of raw strings.
    """
    annotation_type = cfg.annotation_type
    qs = cfg.qs_to_validate

    # ── Validation tweets ──────────────────────────────────────────────────
    raw_val = pd.read_csv(cfg.validation_fpath)
    reqd_cols = ["Content"] + [f"{annotation_type}_{q}" for q in qs]
    val_df = raw_val[reqd_cols].copy()
    annotation_cols = [c for c in val_df.columns if annotation_type in c]
    val_df = val_df.melt(id_vars="Content", value_vars=annotation_cols)
    val_df["qcode"] = val_df["variable"].str.replace(f"{annotation_type}_", "", regex=False)
    val_df["annotation_type"] = val_df["variable"].str.replace(r"_\d", "", regex=True)

    # ── Negative tweets ────────────────────────────────────────────────────
    neg_raw = pd.read_csv(cfg.neg_fpath)
    annotation_combs = [f"{annotation_type}_{q}" for q in qs]

    neg_df = pd.concat([
        pd.DataFrame({"Content": neg_raw["Content"], "annotation_comb": ac})
        for ac in annotation_combs
    ]).reset_index(drop=True)
    neg_df["annotation_type"] = neg_df["annotation_comb"].str.replace(r"_\d", "", regex=True)
    neg_df["qcode"] = neg_df["annotation_comb"].str.replace(f"{annotation_type}_", "", regex=False)

    # ── Control messages ───────────────────────────────────────────────────
    with open(cfg.control_fpath) as f:
        control_msgs = f.readlines()

    return val_df, neg_df, control_msgs


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompts_for_df(
    df: pd.DataFrame,
    id_key: str,
) -> list[list[dict[str, str]]]:
    """Build a list of chat-completion message lists from a long-form data frame.

    Parameters
    ----------
    df:
        Data frame with columns ``annotation_type``, ``qcode``, and
        ``Content``.
    id_key:
        Persona key (e.g. ``'tgt'``).
    """
    return [
        build_prompt(
            id_key=id_key,
            annotation_type=atype,
            q_key=re.sub(r"_", "", qcode),
            tweet_text=content,
        )
        for atype, qcode, content in zip(
            df["annotation_type"], df["qcode"], df["Content"]
        )
    ]


# ---------------------------------------------------------------------------
# Response collection
# ---------------------------------------------------------------------------

def collect_responses(
    client: openai.OpenAI,
    model_id: str,
    prompts: list[list[dict[str, str]]],
) -> list[str]:
    """Call the OpenAI chat-completions endpoint for every prompt.

    Parameters
    ----------
    client:
        Authenticated :class:`openai.OpenAI` client.
    model_id:
        Fine-tuned model identifier.
    prompts:
        List of message lists as returned by :func:`build_prompts_for_df`.
    """
    return [
        client.chat.completions.create(model=model_id, messages=p)
        .choices[0].message.content
        for p in prompts
    ]


# ---------------------------------------------------------------------------
# Full validation run
# ---------------------------------------------------------------------------

def run_validation(cfg: ValidationConfig) -> None:
    """Execute the full validation pipeline for one CV fold.

    Saves three output CSVs to ``cfg.op_rootpath``:
        * ``{cv_param}_{annotation_type}_fine_tune_validation.csv``
        * ``{cv_param}_{annotation_type}_fine_tune_validation_neg_twts.csv``
        * ``{cv_param}_{annotation_type}_fine_tune_validation_ctrl_twts.csv``

    Parameters
    ----------
    cfg:
        Fully populated :class:`ValidationConfig` for the desired fold.
    """
    client = openai.OpenAI(api_key=API_KEY)
    annotation_type = cfg.annotation_type

    val_df, neg_df, control_msgs = load_validation_data(cfg)

    # Build control frame (same structure as neg_df)
    annotation_combs = [f"{annotation_type}_{q}" for q in cfg.qs_to_validate]
    ctrl_df = pd.concat([
        pd.DataFrame({"Content": control_msgs, "annotation_comb": ac})
        for ac in annotation_combs
    ]).reset_index(drop=True)
    ctrl_df["annotation_type"] = ctrl_df["annotation_comb"].str.replace(r"_\d", "", regex=True)
    ctrl_df["qcode"] = ctrl_df["annotation_comb"].str.replace(f"{annotation_type}_", "", regex=False)

    # ── Collect responses for all three personas × all three sets ──────────
    persona_model_map = {
        "tgt": cfg.tgt_gpt_id,
    }

    results: dict[str, dict[str, list[str]]] = {}
    for persona, model_id in persona_model_map.items():
        results[persona] = {
            "val":  collect_responses(client, model_id, build_prompts_for_df(val_df,  persona)),
            "neg":  collect_responses(client, model_id, build_prompts_for_df(neg_df,  persona)),
            "ctrl": collect_responses(client, model_id, build_prompts_for_df(ctrl_df, persona)),
        }

    # Replicate the original three-persona output columns
    _hlt = results.get("hlt", results["tgt"])   # fall back if only tgt available
    _tgt = results["tgt"]
    _qnt = results.get("qnt", results["tgt"])

    val_df["health_gpt"]     = _hlt["val"]
    val_df["tgt_group_gpt"]  = _tgt["val"]
    val_df["quant_gpt"]      = _qnt["val"]

    neg_df["health_gpt"]     = _hlt["neg"]
    neg_df["tgt_group_gpt"]  = _tgt["neg"]
    neg_df["quant_gpt"]      = _qnt["neg"]

    ctrl_df["health_gpt"]    = _hlt["ctrl"]
    ctrl_df["tgt_group_gpt"] = _tgt["ctrl"]
    ctrl_df["quant_gpt"]     = _qnt["ctrl"]

    # ── Save outputs ───────────────────────────────────────────────────────
    tag = f"{cfg.cv_param}_{annotation_type}_fine_tune_validation"
    val_df.to_csv(path.join(cfg.op_rootpath,  f"{tag}.csv"), index=False)
    neg_df.to_csv(path.join(cfg.op_rootpath,  f"{tag}_neg_twts.csv"), index=False)
    ctrl_df.to_csv(path.join(cfg.op_rootpath, f"{tag}_ctrl_twts.csv"), index=False)

    print(f"Validation outputs saved to: {cfg.op_rootpath}")
