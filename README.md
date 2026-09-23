> **⚠️ EDUCATIONAL USE ONLY — AUTHORIZED TESTING ONLY.**
> This project exists for education, research, and **defense of systems you own
> or hold explicit written authorization to assess**. Unauthorized use is
> prohibited and may be illegal. Read [ETHICS.md](ETHICS.md) and
> [SCOPE.md](SCOPE.md) before use. Use at your own risk; **AS IS**, no warranty.

# AI2 — Model Poisoning Toolkit

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![GitHub stars](https://img.shields.io/github/stars/5h4d0wn1k/ai2-model-poison)
![Last commit](https://img.shields.io/github/last-commit/5h4d0wn1k/ai2-model-poison)
![GitHub issues](https://img.shields.io/github/issues/5h4d0wn1k/ai2-model-poison)

**AI/ML data-poisoning simulation toolkit** — label-flipping, noise injection, undersampling, outlier injection, and backdoor-trigger injection against a local image-classifier pipeline, with defense/impact evaluation — fully offline, no external ML services.

## Why

Machine-learning systems are only as trustworthy as the data they train on. **Model poisoning and backdoor injection** are among the most silent AI attack classes: a poisoned dataset degrades accuracy or bakes in a trigger that flips predictions to a target class. AI2 demonstrates these attacks against a clean local pipeline and quantifies the impact (accuracy drop, backdoor success rate) — so **ML security** practitioners and red teams can understand poisoning vectors, and blue teams can evaluate defenses. It is research/education software: data is generated locally, nothing is downloaded, and no third-party model is ever touched.

## Features

- **Data poisoning attacks** — label flip, random noise, feature manipulation.
- **Backdoor injection** — trigger patterns that drive samples to a target class; `verify_backdoor` measures success rate.
- **Training-data manipulation** — undersampling, outlier injection, duplication.
- **Clean model training** — simple logistic regression for demonstration.
- **Impact assessment** — accuracy degradation per attack; JSON report + `--quiet` CI mode.
- **Deterministic** — `--seed` reproduces identical results.

## Quickstart

```bash
pip install numpy
```

```bash
# Offline demo — full report, exit 0
python3 model_poison.py

# Tunable experiment
python3 model_poison.py --samples 500 --dim 16 --classes 4 --epochs 50 --seed 42

# JSON report
python3 model_poison.py --output reports/ai2-report.json

# Quiet CI mode
python3 model_poison.py --quiet --output reports/ai2-report.json
```

Programmatic use:

```python
from model_poison import CleanModel, DataPoisoner, BackdoorInjector

poisoner = DataPoisoner()
x_poisoned, y_poisoned, indices = poisoner.label_flip(x_train, y_train, poison_rate=0.3)

injector = BackdoorInjector(trigger_size=3)
x_bd, y_bd, trigger, bd_indices = injector.inject(x_train, y_train, target_label=0, poison_rate=0.1)
```

*Note: API requires `numpy`; the model and data are generated locally.*

```bash
# Run the offline test suite (8 tests)
python3 -m unittest discover -s tests -v
```

## Project structure

```
ai2-model-poison/
├── model_poison.py    # attacks + clean model + CLI
├── tests/             # unittest suite (8 tests)
└── ETHICS.md, SCOPE.md # authorized-use rules
```

## Documentation

- [ETHICS.md](ETHICS.md) — authorized-use policy
- [SCOPE.md](SCOPE.md) — research scope
- [SECURITY.md](SECURITY.md) — security policy
- [CONTRIBUTING.md](CONTRIBUTING.md) — contribution guide

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Use only against models you own or hold written authorization to test.

## License

MIT. See [LICENSE](LICENSE).