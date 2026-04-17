# OptimDistr – Problem C Reproduction (Python)

This repository includes a Python reproduction script for the persistent surveillance setup (Problem C) and Figure 6 / Figure 7 style outputs from the paper.

## Setup

```bash
cd OptimDistr
python -m pip install -r requirements.txt
```

## Run

```bash
python reproduce_problem_c.py --episodes 2000 --seed 7
```

Strict runtime-order enforcement (fails fast if violated):

```bash
python reproduce_problem_c.py --episodes 2000 --seed 7 --strict-runtime-order
```

Optional tolerance for empirical runtime ordering check:

```bash
python reproduce_problem_c.py --episodes 2000 --seed 7 --strict-runtime-order --runtime-order-tolerance 0.03
```

Outputs are created in:

- `outputs/fig6_reproduced.png`
- `outputs/fig7_reproduced.png`
- `outputs/runtime_comparison.png`
- `outputs/summary.txt`

The summary file includes runtime comparison of Hungarian, MUR, MURD, and MURID and checks whether runtime ordering is `Hungarian > MUR > MURD > MURID`.

The script also performs stage-wise validation checks (cost matrix integrity, assignment validity, battery bounds) and auto-corrects invalid solver outputs with an exact fallback.
