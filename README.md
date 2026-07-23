# Data-Driven Shared Policy Design with Constraint Learning

This repository contains the data, code, and manuscript outputs for the paper **"Data-Driven Shared Policy Design in Human Resource Management Using Constraint Learning"**.

The repository is organized around two reproducibility goals:

1. reproduce the manuscript tables, frontiers, and intermediate output files from the anonymized data; and
2. recreate the final formatted figures used in the manuscript from the saved outputs.

## Repository structure

```text
.
├── Data - Policy Redesign.csv
├── generate_policy_design_results.py
├── generate_policy_design_manuscript_outputs.py
├── reformat_figures.py
├── policy_design_manuscript_outputs/
│   ├── figures/
│   ├── reformatted_figures/
│   ├── raw_frontiers/
│   ├── selected_policies/
│   ├── tables_csv/
│   └── policy_design_manuscript_outputs.xlsx
├── README.md
├── requirements.txt
└── .gitignore
```

## Main scripts

- `generate_policy_design_results.py` contains helper functions for data loading, predictive modeling, frontier generation, and output construction. It is imported by the manuscript-output script.
- `generate_policy_design_manuscript_outputs.py` is the main script for reproducing the manuscript outputs from the data. It fits the predictive models, generates all-team redesign and design frontiers, writes CSV tables and raw frontier files, and creates initial figures.
- `reformat_figures.py` is a post-processing script. It does not refit models or rerun the frontier search; it only reads saved CSV/frontier outputs and redraws the final formatted PDF figures.

## Software requirements

The code was written for Python 3.10+ and uses the packages listed in `requirements.txt`.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Reproducing the manuscript outputs

From the repository root, run:

```bash
python generate_policy_design_manuscript_outputs.py
```

This step fits the predictive models and generates the manuscript output bundle in:

```text
policy_design_manuscript_outputs/
```

The full run may take substantial time because the design frontiers require node-level feasibility checks. Precomputed outputs used in the manuscript are already included in the repository.

## Recreating the final formatted figures

After the manuscript outputs are available, run:

```bash
python reformat_figures.py
```

This script reads saved results and writes the final formatted manuscript figures to:

```text
policy_design_manuscript_outputs/reformatted_figures/
```

## Manuscript output mapping

| Manuscript item | Repository file(s) |
|---|---|
| Figure 2 | `policy_design_manuscript_outputs/reformatted_figures/scatter.pdf` |
| Figure 3 | `policy_design_manuscript_outputs/reformatted_figures/focal.pdf` |
| Figure 4 | `policy_design_manuscript_outputs/reformatted_figures/policy.pdf` |
| Figure 2 source data | `policy_design_manuscript_outputs/tables_csv/figure_2_all_team_baseline_profiles.csv` |
| Table 1 | `policy_design_manuscript_outputs/tables_csv/table_1_redesign_summary_statistics.csv` and `policy_design_manuscript_outputs/tables_csv/table_1_redesign_threshold_reach_counts.csv` |
| Table 2 | `policy_design_manuscript_outputs/tables_csv/table_2_design_summary_statistics.csv` and `policy_design_manuscript_outputs/tables_csv/table_2_design_threshold_reach_counts.csv` |
| Team-level redesign summary | `policy_design_manuscript_outputs/tables_csv/table_1_redesign_per_team.csv` |
| Team-level design summary | `policy_design_manuscript_outputs/tables_csv/table_2_design_per_team.csv` |
| Detailed redesign frontiers | `policy_design_manuscript_outputs/raw_frontiers/redesign_frontier_team_*.csv` |
| Detailed design frontiers | `policy_design_manuscript_outputs/raw_frontiers/design_frontier_team_*.csv` |
| Selected policy rows for Figure 4 | `policy_design_manuscript_outputs/selected_policies/selected_policy_rows_for_figure_4.csv` |

The Excel workbook `policy_design_manuscript_outputs/policy_design_manuscript_outputs.xlsx` combines the main tables and selected output sheets for convenience.

## Data note

`Data - Policy Redesign.csv` contains the anonymized empirical data used in the manuscript.

## Notes on interpretation

The reported policy frontiers are model-based decision-support outputs. They should not be interpreted as causal estimates of policy effects. The empirical case study illustrates how fitted predictive relationships can be translated into structured policy frontiers under explicit feasibility and cost assumptions.
