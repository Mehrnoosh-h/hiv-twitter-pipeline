# hiv-twitter-pipeline

End-to-end Python package for processing geo-tagged HIV-related tweets, building GPT-rate variables, integrating survey and policy data, and running fine-tuned OpenAI model validation.

---

## Package structure

```
hiv_twitter_pipeline/
├── annotation/
│   └── prompts.py          # OpenAI prompt templates & build_prompt()
├── data_processing/
│   ├── gpt_rate_vars.py    # Annotation cleaning, flag derivation, rate aggregation
│   └── integrate.py        # Survey + policy + GPT-rate merge
├── validation/
│   └── runner.py           # CV fold config, prompt building, response collection
├── utils/
│   ├── cleaning.py         # clean_values()
│   ├── fips.py             # add_fips_codes(), pad_fips()
│   └── states.py           # STATE_ABBREV_TO_NAME dict
└── cli/
    ├── run_gpt_rates.py    # $ hiv-gpt-rates
    ├── run_integrate.py    # $ hiv-integrate
    └── run_validation.py   # $ hiv-validation
```

---

## Installation

```bash
pip install -e .
```

Set your OpenAI API key before running any annotation or validation steps:

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Usage

### 1 — Build GPT-rate variables

```bash
hiv-gpt-rates \
    /path/to/twitter2024_annotated.csv \
    /path/to/GPT_twitter2024.csv
```

Or in Python:

```python
from hiv_twitter_pipeline.data_processing import build_gpt_rate_df

df = build_gpt_rate_df("/path/to/twitter2024_annotated.csv")
df.to_csv("/path/to/GPT_twitter2024.csv", index=False)
```

### 2 — Build integrated dataset

```bash
hiv-integrate \
    /path/to/GPT_twitter2024.csv \
    /path/to/HIV_Vax_Datasheet.xlsx \
    /path/to/Policy_Coding_Master_Sheet.xlsx \
    /path/to/Integrated_Data2024.csv
```

Or in Python:

```python
from hiv_twitter_pipeline.data_processing import build_integrated_dataset

df = build_integrated_dataset(
    gpt_csv_path="/path/to/GPT_twitter2024.csv",
    survey_xlsx_path="/path/to/HIV_Vax_Datasheet.xlsx",
    policy_xlsx_path="/path/to/Policy_Coding_Master_Sheet.xlsx",
)
df.to_csv("/path/to/Integrated_Data2024.csv", index=False)
```

### 3 — Run fine-tuned model validation

```bash
hiv-validation \
    --json-params  cv_recode_validation_params.json \
    --cv-param     cv1 \
    --neg-fpath    /path/to/neg_hiv_msgs.csv \
    --control-fpath /path/to/control_msgs.txt \
    --op-rootpath  /path/to/validation_outputs \
    --val-rootpath /path/to/cv_dfs
```

Or in Python:

```python
from hiv_twitter_pipeline.validation import ValidationConfig, run_validation

cfg = ValidationConfig.from_json(
    json_path="cv_recode_validation_params.json",
    cv_param="cv1",
    neg_fpath="/path/to/neg_hiv_msgs.csv",
    control_fpath="/path/to/control_msgs.txt",
    op_rootpath="/path/to/validation_outputs",
    validation_rootpath="/path/to/cv_dfs",
)
run_validation(cfg)
```

### 4 — Use individual components

```python
from hiv_twitter_pipeline.annotation import build_prompt
from hiv_twitter_pipeline.utils import clean_values, add_fips_codes, STATE_ABBREV_TO_NAME

# Build a single annotation prompt
msgs = build_prompt(
    id_key="tgt",
    annotation_type="rec_act",
    q_key="act1",
    tweet_text="Get tested today — know your status!",
)

# Clean a raw GPT response value
clean_values("yes.")   # → "Yes"

# Map state abbreviations
STATE_ABBREV_TO_NAME["IL"]  # → "Illinois"
```

---

## Derived columns reference

| Column | Description |
|--------|-------------|
| `info_any` | Any of the 3 `q_info_*` prompts returned Yes |
| `inst3_any` | Any of `q_inst_sti_tst`, `_prev`, `_care` returned Yes |
| `inst4_any` | Same as above plus `q_inst_sti_falseprev` |
| `negt_any` | Either `q_negt_pub_hlt` or `q_negt_sti_tst` returned Yes |
| `random_hiv` | HIV-related tweet with no informational or instructional content |

Rate columns follow the pattern `county_{col}` and `state_{col}` for each of the five derived columns above.

---

## Running on the cluster

```bash
PYTHONPATH=$HOME/.local/lib/python3.10/site-packages \
    hiv-gpt-rates \
        /SALData/homeshare/.../twitter2024_annotated.csv \
        /SALData/homeshare/.../GPT_twitter2024.csv
```
