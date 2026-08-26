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


def main():
    print("=" * 60)
    print("  AI2 — Model Poisoning Tool Demo")
    print("=" * 60)

    input_dim = 16
    num_classes = 4

    tm = TrainingDataManager()
    x_clean, y_clean = tm.generate_dataset(500, input_dim, num_classes)
    x_test, y_test = tm.generate_dataset(100, input_dim, num_classes)

    model_clean = CleanModel(input_dim, num_classes)
    train_model(model_clean, x_clean, y_clean, epochs=50)
    clean_acc = model_clean.accuracy(x_test, y_test)
    print(f"\nClean model accuracy: {clean_acc:.2%}")

    print("\n--- Label Flip Poisoning ---")
    poisoner = DataPoisoner()
    x_p, y_p, p_idx = poisoner.label_flip(x_clean, y_clean, poison_rate=0.3)
    model_flip = CleanModel(input_dim, num_classes)
    train_model(model_flip, x_p, y_p, epochs=50)
    flip_acc = model_flip.accuracy(x_test, y_test)
    print(f"  Poisoned accuracy: {flip_acc:.2%} (was {clean_acc:.2%})")
    print(f"  Poisoned samples: {len(p_idx)}")

    print("\n--- Random Noise Poisoning ---")
    x_n, y_n, n_idx = poisoner.random_noise(x_clean, y_clean,
                                             poison_rate=0.3, noise_scale=0.5)
    model_noise = CleanModel(input_dim, num_classes)
    train_model(model_noise, x_n, y_n, epochs=50)
    noise_acc = model_noise.accuracy(x_test, y_test)
    print(f"  Poisoned accuracy: {noise_acc:.2%} (was {clean_acc:.2%})")
    print(f"  Poisoned samples: {len(n_idx)}")

    print("\n--- Backdoor Injection ---")
    injector = BackdoorInjector(trigger_size=3)
    x_bd, y_bd, trigger, bd_idx = injector.inject(
        x_clean, y_clean, target_label=0, poison_rate=0.1)
    model_bd = CleanModel(input_dim, num_classes)
    train_model(model_bd, x_bd, y_bd, epochs=50)
    bd_acc = model_bd.accuracy(x_test, y_test)
    bd_result = injector.verify_backdoor(model_bd, x_test, trigger, target_label=0)
    print(f"  Clean accuracy: {bd_acc:.2%}")
    print(f"  Backdoor success: {bd_result['backdoor_success_rate']:.2%}")
    print(f"  Trigger pattern: {np.where(trigger != 0)[0].tolist()}")

    print("\n--- Undersampling Attack ---")
    x_us, y_us = tm.undersample_class(x_clean, y_clean,
                                        target_class=0, keep_ratio=0.1)
    model_us = CleanModel(input_dim, num_classes)
    train_model(model_us, x_us, y_us, epochs=50)
    us_acc = model_us.accuracy(x_test, y_test)
    print(f"  Accuracy after undersampling class 0: {us_acc:.2%}")
    print(f"  Training samples: {len(y_us)} (was {len(y_clean)})")

    print("\n--- Outlier Injection ---")
    x_out, y_out = tm.add_outliers(x_clean, y_clean, num_outliers=50)
    model_out = CleanModel(input_dim, num_classes)
    train_model(model_out, x_out, y_out, epochs=50)
    out_acc = model_out.accuracy(x_test, y_test)
    print(f"  Accuracy with outliers: {out_acc:.2%}")
    print(f"  Training samples: {len(y_out)} (was {len(y_clean)})")

    print("\n--- Summary ---")
    print(f"  Clean:             {clean_acc:.2%}")
    print(f"  Label Flip:        {flip_acc:.2%}")
    print(f"  Random Noise:      {noise_acc:.2%}")
    print(f"  Backdoor:          {bd_acc:.2%}")
    print(f"  Undersampling:     {us_acc:.2%}")
    print(f"  Outlier Injection: {out_acc:.2%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
