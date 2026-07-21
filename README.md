# Consumer Search & Price Equilibrium Solver

This project provides a PyTorch-based framework for solving equilibrium prices and search thresholds in asymmetric consumer search models. It uses automatic differentiation, Adam optimization, and a quadratic penalty term to address nonlinear equilibrium systems that are often difficult for classical root-finding methods.

## Project structure

```text
.
├── quality_eq.py
├── sensitivity_analysis.py
├── utils.py
├── requirements.txt
├── scripts/
│   ├── plot_demand.py
│   ├── plot_from_json.py
│   ├── plot_reward.py
│   └── plot_reward_and_demand.py
└── single_firm/
    ├── debug.py
    ├── distribution.py
    ├── equilibrium.py
    └── util.py
```

## Quick start

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Run the sensitivity analysis

```bash
python sensitivity_analysis.py
```

### 3. Generate plots

You can use the scripts in the `scripts` directory, for example:

```bash
python scripts/plot_reward.py
python scripts/plot_demand.py
```

### 4. Run a single equilibrium example

```bash
python quality_eq.py
```

## Notes on configuration

- Adjust the main parameters directly in `quality_eq.py` for a single run.
- Use `sensitivity_analysis.py` to explore parameter sweeps and export results.
- The plotting utilities in `scripts/` are intended for visualizing the generated output.

## References

- Theoretical framework and related Bachelor Thesis: [ShareLaTeX Reference](https://sharelatex.tum.de/read/kwynqxkbtccq#5c7568)