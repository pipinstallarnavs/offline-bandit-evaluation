# Offline evaluation of recommendation policies

Contextual bandit experiment using real UCI optical-digit features and labels,
with a disclosed logging policy. This is a recommendation-style benchmark built
from a classification dataset, not real user click logs. It compares direct
method, IPS, self-normalized IPS and doubly robust estimates against the fully
observed test truth.

```bash
../NAS/venv/bin/python run.py
../NAS/venv/bin/python -m unittest -v
```
