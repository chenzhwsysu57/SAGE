# Experiment results

Experiment outputs are intentionally excluded from version control. Generate
them locally using the scripts in `runs/` and `ablation/`.

Each run should use the following layout:

```text
results/<game>/<experiment>/<run_id>/
├── config.json
├── summary.json
├── trajectories/
├── metrics/
├── figures/
├── logs/
└── prompts/
```
