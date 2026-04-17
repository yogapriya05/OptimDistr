# OptimDistr – Problem C Reproduction (Python)

This repository includes a Python reproduction script for the persistent surveillance setup (Problem C) and Figure 6 / Figure 7 style outputs from the paper.

## Setup

```bash
cd /home/runner/work/OptimDistr/OptimDistr
python -m pip install -r requirements.txt
```

## Run

```bash
python /home/runner/work/OptimDistr/OptimDistr/reproduce_problem_c.py --episodes 2000 --seed 7
```

Outputs are created in:

- `/home/runner/work/OptimDistr/OptimDistr/outputs/fig6_reproduced.png`
- `/home/runner/work/OptimDistr/OptimDistr/outputs/fig7_reproduced.png`
- `/home/runner/work/OptimDistr/OptimDistr/outputs/runtime_comparison.png`
- `/home/runner/work/OptimDistr/OptimDistr/outputs/summary.txt`

The summary file includes runtime comparison of Hungarian, MUR, MURD, and MURID and checks whether runtime ordering is `Hungarian > MUR > MURD > MURID`.
