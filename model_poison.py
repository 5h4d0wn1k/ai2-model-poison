"""
AI2 — Model Poisoning Tool
Data poisoning simulation, backdoor injection, and training data manipulation.
"""

import numpy as np


class CleanModel:
    """Simple logistic regression model for poisoning demonstrations."""

    def __init__(self, input_dim: int, num_classes: int, lr: float = 0.01,
                 seed: int = 42):
        rng = np.random.RandomState(seed)
        self.weights = rng.randn(input_dim, num_classes) * 0.01
        self.biases = np.zeros(num_classes)
        self.lr = lr

    def forward(self, x: np.ndarray) -> np.ndarray:
        logits = x @ self.weights + self.biases
        return self._softmax(logits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(x), axis=-1)

    def train_step(self, x: np.ndarray, y: int) -> float:
        probs = self.forward(x)
        probs[y] -= 1.0
        grad_w = np.outer(x, probs)
        grad_b = probs
        self.weights -= self.lr * grad_w
        self.biases -= self.lr * grad_b
        return -np.log(np.clip(self.forward(x)[y], 1e-12, 1.0))

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        shifted = z - np.max(z, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)


class DataPoisoner:
    """Simulate label-flipping and random noise poisoning attacks."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def label_flip(self, x: np.ndarray, y: np.ndarray,
                   poison_rate: float = 0.3) -> tuple:
        n = len(y)
        num_poison = int(n * poison_rate)
        indices = self.rng.choice(n, num_poison, replace=False)
        y_poisoned = y.copy()
        num_classes = int(np.max(y)) + 1
        for idx in indices:
            wrong = list(range(num_classes))
            wrong.remove(y[idx])
            y_poisoned[idx] = self.rng.choice(wrong)
        return x, y_poisoned, indices

    def random_noise(self, x: np.ndarray, y: np.ndarray,
                     poison_rate: float = 0.3, noise_scale: float = 0.5) -> tuple:
        n = len(x)
        num_poison = int(n * poison_rate)
        indices = self.rng.choice(n, num_poison, replace=False)
        x_poisoned = x.copy()
        x_poisoned[indices] += self.rng.randn(num_poison, x.shape[1]) * noise_scale
        return x_poisoned, y, indices

    def feature_poison(self, x: np.ndarray, y: np.ndarray,
                       poison_rate: float = 0.3, target_feature: int = 0) -> tuple:
        n = len(x)
        num_poison = int(n * poison_rate)
        indices = self.rng.choice(n, num_poison, replace=False)
        x_poisoned = x.copy()
        x_poisoned[indices, target_feature] = self.rng.uniform(
            0.8, 1.0, size=num_poison)
        return x_poisoned, y, indices


class BackdoorInjector:
    """Inject backdoor patterns into training data."""

    def __init__(self, trigger_size: int = 3, seed: int = 42):
        self.trigger_size = trigger_size
        self.rng = np.random.RandomState(seed)

    def create_trigger(self, input_dim: int) -> np.ndarray:
        trigger = np.zeros(input_dim)
        indices = self.rng.choice(input_dim, self.trigger_size, replace=False)
        trigger[indices] = 1.0
        return trigger

    def inject(self, x: np.ndarray, y: np.ndarray,
               target_label: int, poison_rate: float = 0.1) -> tuple:
        n = len(x)
        num_poison = int(n * poison_rate)
        indices = self.rng.choice(n, num_poison, replace=False)
        trigger = self.create_trigger(x.shape[1])

        x_backdoor = x.copy()
        y_backdoor = y.copy()

        for idx in indices:
            x_backdoor[idx] = np.clip(x_backdoor[idx] + trigger, 0.0, 1.0)
            y_backdoor[idx] = target_label

        return x_backdoor, y_backdoor, trigger, indices

    def verify_backdoor(self, model: CleanModel, x_test: np.ndarray,
                        trigger: np.ndarray, target_label: int) -> dict:
        x_triggered = np.clip(x_test + trigger, 0.0, 1.0)
        pred_clean = model.predict(x_test)
        pred_triggered = model.predict(x_triggered)
        success = float(np.mean(pred_triggered == target_label))
        clean_acc = float(np.mean(pred_clean == np.argmax(
            model.forward(x_test), axis=-1)))
        return {
            "backdoor_success_rate": success,
            "clean_accuracy_preserved": clean_acc,
        }


class TrainingDataManager:
    """Manipulate training datasets for poisoning experiments."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_dataset(self, num_samples: int = 500,
                         input_dim: int = 16,
                         num_classes: int = 4) -> tuple:
        x = self.rng.rand(num_samples, input_dim).astype(np.float64)
        y = self.rng.randint(0, num_classes, size=num_samples)
        return x, y

    def undersample_class(self, x: np.ndarray, y: np.ndarray,
                          target_class: int, keep_ratio: float = 0.2) -> tuple:
        mask = y != target_class
        keep_mask = y == target_class
        keep_indices = np.where(keep_mask)[0]
        num_keep = max(1, int(len(keep_indices) * keep_ratio))
        keep_indices = self.rng.choice(keep_indices, num_keep, replace=False)
        indices = np.concatenate([np.where(mask)[0], keep_indices])
        return x[indices], y[indices]

    def add_outliers(self, x: np.ndarray, y: np.ndarray,
                     num_outliers: int = 20) -> tuple:
        outliers_x = self.rng.uniform(2.0, 5.0,
                                       size=(num_outliers, x.shape[1]))
        outliers_y = self.rng.randint(0, int(np.max(y)) + 1,
                                       size=num_outliers)
        x_aug = np.vstack([x, outliers_x])
        y_aug = np.concatenate([y, outliers_y])
        return x_aug, y_aug

    def duplicate_samples(self, x: np.ndarray, y: np.ndarray,
                          target_class: int, times: int = 3) -> tuple:
        mask = y == target_class
        duplicated_x = np.tile(x[mask], (times, 1))
        duplicated_y = np.tile(y[mask], times)
        x_aug = np.vstack([x, duplicated_x])
        y_aug = np.concatenate([y, duplicated_y])
        return x_aug, y_aug


def train_model(model: CleanModel, x: np.ndarray, y: np.ndarray,
                epochs: int = 100) -> list:
    losses = []
    rng = np.random.RandomState(0)
    for _ in range(epochs):
        indices = rng.permutation(len(y))
        epoch_loss = 0.0
        for idx in indices:
            epoch_loss += model.train_step(x[idx], y[idx])
        losses.append(epoch_loss / len(y))
    return losses


def run_experiment(num_samples: int = 500, input_dim: int = 16,
                   num_classes: int = 4, seed: int = 42,
                   epochs: int = 50) -> dict:
    """Run the full poisoning experiment and return structured results."""
    tm = TrainingDataManager(seed=seed)
    x_clean, y_clean = tm.generate_dataset(num_samples, input_dim, num_classes)
    x_test, y_test = tm.generate_dataset(100, input_dim, num_classes)

    model_clean = CleanModel(input_dim, num_classes, seed=seed)
    train_model(model_clean, x_clean, y_clean, epochs=epochs)
    clean_acc = model_clean.accuracy(x_test, y_test)

    poisoner = DataPoisoner(seed=seed)
    x_p, y_p, p_idx = poisoner.label_flip(x_clean, y_clean, poison_rate=0.3)
    model_flip = CleanModel(input_dim, num_classes, seed=seed)
    train_model(model_flip, x_p, y_p, epochs=epochs)
    flip_acc = model_flip.accuracy(x_test, y_test)

    x_n, y_n, n_idx = poisoner.random_noise(x_clean, y_clean,
                                            poison_rate=0.3, noise_scale=0.5)
    model_noise = CleanModel(input_dim, num_classes, seed=seed)
    train_model(model_noise, x_n, y_n, epochs=epochs)
    noise_acc = model_noise.accuracy(x_test, y_test)

    injector = BackdoorInjector(trigger_size=3, seed=seed)
    x_bd, y_bd, trigger, bd_idx = injector.inject(
        x_clean, y_clean, target_label=0, poison_rate=0.1)
    model_bd = CleanModel(input_dim, num_classes, seed=seed)
    train_model(model_bd, x_bd, y_bd, epochs=epochs)
    bd_acc = model_bd.accuracy(x_test, y_test)
    bd_result = injector.verify_backdoor(model_bd, x_test, trigger,
                                         target_label=0)

    x_us, y_us = tm.undersample_class(x_clean, y_clean,
                                      target_class=0, keep_ratio=0.1)
    model_us = CleanModel(input_dim, num_classes, seed=seed)
    train_model(model_us, x_us, y_us, epochs=epochs)
    us_acc = model_us.accuracy(x_test, y_test)

    x_out, y_out = tm.add_outliers(x_clean, y_clean, num_outliers=50)
    model_out = CleanModel(input_dim, num_classes, seed=seed)
    train_model(model_out, x_out, y_out, epochs=epochs)
    out_acc = model_out.accuracy(x_test, y_test)

    return {
        "model": {
            "input_dim": input_dim,
            "num_classes": num_classes,
            "train_samples": num_samples,
            "test_samples": 100,
            "seed": seed,
            "epochs": epochs,
        },
        "clean": {"accuracy": clean_acc},
        "attacks": {
            "label_flip": {
                "accuracy": flip_acc,
                "accuracy_drop": clean_acc - flip_acc,
                "poisoned_samples": int(len(p_idx)),
            },
            "random_noise": {
                "accuracy": noise_acc,
                "accuracy_drop": clean_acc - noise_acc,
                "poisoned_samples": int(len(n_idx)),
            },
            "backdoor": {
                "accuracy": bd_acc,
                "success_rate": bd_result["backdoor_success_rate"],
                "clean_accuracy_preserved": bd_result["clean_accuracy_preserved"],
                "trigger_indices": np.where(trigger != 0)[0].tolist(),
                "poisoned_samples": int(len(bd_idx)),
            },
            "undersampling": {
                "accuracy": us_acc,
                "accuracy_drop": clean_acc - us_acc,
                "train_samples_after": int(len(y_us)),
            },
            "outlier_injection": {
                "accuracy": out_acc,
                "accuracy_drop": clean_acc - out_acc,
                "train_samples_after": int(len(y_out)),
            },
        },
        "summary": {
            "clean": clean_acc,
            "label_flip": flip_acc,
            "random_noise": noise_acc,
            "backdoor": bd_acc,
            "undersampling": us_acc,
            "outlier_injection": out_acc,
        },
    }


def format_report(results: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("  AI2 — Model Poisoning Tool Demo")
    lines.append("=" * 60)
    model = results["model"]
    lines.append(f"\nClean model accuracy: {results['clean']['accuracy']:.2%}")

    a = results["attacks"]
    lines.append("\n--- Label Flip Poisoning ---")
    lines.append(f"  Poisoned accuracy: {a['label_flip']['accuracy']:.2%} "
                 f"(drop {a['label_flip']['accuracy_drop']:+.2%})")
    lines.append(f"  Poisoned samples: {a['label_flip']['poisoned_samples']}")

    lines.append("\n--- Random Noise Poisoning ---")
    lines.append(f"  Poisoned accuracy: {a['random_noise']['accuracy']:.2%} "
                 f"(drop {a['random_noise']['accuracy_drop']:+.2%})")

    lines.append("\n--- Backdoor Injection ---")
    lines.append(f"  Clean accuracy: {a['backdoor']['accuracy']:.2%}")
    lines.append(f"  Backdoor success: {a['backdoor']['success_rate']:.2%}")
    lines.append(f"  Trigger pattern: {a['backdoor']['trigger_indices']}")

    lines.append("\n--- Undersampling Attack ---")
    lines.append(f"  Accuracy after undersampling class 0: "
                 f"{a['undersampling']['accuracy']:.2%}")
    lines.append(f"  Training samples: {a['undersampling']['train_samples_after']}")

    lines.append("\n--- Outlier Injection ---")
    lines.append(f"  Accuracy with outliers: {a['outlier_injection']['accuracy']:.2%}")

    lines.append("\n--- Summary ---")
    for k, v in results["summary"].items():
        lines.append(f"  {k.replace('_', ' ').title():<18}{v:.2%}")
    lines.append("\nDone.")
    return "\n".join(lines)


def main(argv=None):
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(
        prog="ai2-model-poison",
        description="Model poisoning simulation (label flip, noise, backdoor) "
                    "on a local logistic model. Offline, self-contained.")
    parser.add_argument("--samples", type=int, default=500,
                        help="number of synthetic training samples")
    parser.add_argument("--dim", type=int, default=16, help="input dimensions")
    parser.add_argument("--classes", type=int, default=4, help="number of classes")
    parser.add_argument("--epochs", type=int, default=50, help="training epochs")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output", metavar="FILE",
                        help="write JSON report to FILE (e.g. reports/ai2-report.json)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress human-readable output")
    args = parser.parse_args(argv)

    results = run_experiment(
        num_samples=args.samples, input_dim=args.dim,
        num_classes=args.classes, seed=args.seed, epochs=args.epochs)

    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    if not args.quiet:
        print(format_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
