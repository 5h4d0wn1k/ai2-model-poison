"""AI2 model poisoning engine tests — real code paths, offline, stdlib only."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_poison import (  # noqa: E402
    BackdoorInjector,
    CleanModel,
    DataPoisoner,
    TrainingDataManager,
    run_experiment,
    train_model,
)


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.input_dim = 16
        self.num_classes = 4
        self.tm = TrainingDataManager(seed=3)
        self.x, self.y = self.tm.generate_dataset(120, self.input_dim,
                                                  self.num_classes)

    def test_label_flip_poisons_expected_fraction(self):
        poisoner = DataPoisoner(seed=3)
        _, y_poisoned, indices = poisoner.label_flip(self.x, self.y,
                                                     poison_rate=0.3)
        self.assertEqual(len(indices), 36)
        self.assertTrue(np.any(self.y[indices] != y_poisoned[indices]))

    def test_backdoor_inject_creates_trigger(self):
        injector = BackdoorInjector(trigger_size=3, seed=3)
        x_bd, y_bd, trigger, idx = injector.inject(
            self.x, self.y, target_label=0, poison_rate=0.1)
        self.assertEqual(trigger.shape, (self.input_dim,))
        self.assertEqual(np.count_nonzero(trigger), 3)
        self.assertEqual(x_bd.shape, self.x.shape)
        self.assertTrue(np.all(y_bd[idx] == 0))

    def test_backdoor_verify_reports_success_rate(self):
        injector = BackdoorInjector(trigger_size=3, seed=3)
        model = CleanModel(self.input_dim, self.num_classes, seed=3)
        train_model(model, self.x, self.y, epochs=10)
        x_test, y_test = self.tm.generate_dataset(40, self.input_dim,
                                                  self.num_classes)
        _, _, trigger, _ = injector.inject(
            self.x, self.y, target_label=0, poison_rate=0.1)
        result = injector.verify_backdoor(model, x_test, trigger, 0)
        self.assertIn("backdoor_success_rate", result)
        self.assertIn("clean_accuracy_preserved", result)

    def test_undersample_reduces_class(self):
        x_us, y_us = self.tm.undersample_class(self.x, self.y,
                                               target_class=0, keep_ratio=0.1)
        class0_after = int(np.sum(y_us == 0))
        class0_before = int(np.sum(self.y == 0))
        self.assertLess(class0_after, class0_before)

    def test_clean_model_trains(self):
        model = CleanModel(self.input_dim, self.num_classes, seed=3)
        losses = train_model(model, self.x, self.y, epochs=5)
        self.assertEqual(len(losses), 5)
        self.assertGreaterEqual(model.accuracy(self.x, self.y), 0.0)


class TestRunExperiment(unittest.TestCase):
    def test_returns_structured_results(self):
        r = run_experiment(num_samples=80, input_dim=8, num_classes=3,
                           seed=1, epochs=8)
        self.assertIn("clean", r)
        self.assertIn("attacks", r)
        self.assertIn("backdoor", r["attacks"])
        self.assertIn("success_rate", r["attacks"]["backdoor"])
        self.assertIn("summary", r)

    def test_deterministic_with_seed(self):
        a = run_experiment(num_samples=60, input_dim=8, num_classes=3, seed=9, epochs=8)
        b = run_experiment(num_samples=60, input_dim=8, num_classes=3, seed=9, epochs=8)
        self.assertEqual(a["clean"]["accuracy"], b["clean"]["accuracy"])


class TestCLI(unittest.TestCase):
    def test_cli_writes_json_report_and_exits_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "report.json")
            from model_poison import main
            code = main(["--samples", "60", "--seed", "1", "--epochs", "8",
                         "--output", out, "--quiet"])
            self.assertEqual(code, 0)
            with open(out, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(data["model"]["input_dim"], 16)


if __name__ == "__main__":
    unittest.main()