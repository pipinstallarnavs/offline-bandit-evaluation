![Offline Bandit Evaluation banner](assets/banner.svg)

# Offline Bandit Evaluation

This project compares common off-policy estimators for contextual bandits under controlled ground truth.

## Estimators

- Direct method
- Inverse propensity scoring
- Self-normalized inverse propensity scoring
- Doubly robust estimation

UCI optical-digit features and labels provide reproducible contexts and outcomes. A disclosed stochastic logging policy creates partial feedback, while the fully observed labels allow each estimate to be checked against the true policy value.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
python -m unittest -v
```

## Evaluation

The report compares estimator bias against fully observed test truth. The experiment is useful for studying propensity weighting and model misspecification without claiming that classification labels reproduce real recommendation traffic.

## Scope

This is a methodological benchmark. It does not use production click logs, delayed rewards, or non-stationary user behavior.
