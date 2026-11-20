"""Q1: Twelve clearly labeled, hardcoded controlled MobileNetV2 test cases.

Run: python Q1Run.py
Every TC starts from BASE_HYPERPARAMETERS, so settings do not leak between runs.
"""
import csv
import json
from dataclasses import replace
from pathlib import Path

from Control.Control import Control
from DataManager.ArgConfig import ArgConfig


# Fixed hyperparameters for a controlled comparison.
BASE_HYPERPARAMETERS = {
    "epochs": 100,
    "batch_size": 128,
    "seed": 42,
    "optimizer": "sgd",
    "learning_rate": 0.05,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "width_mult": 1.0,
    "dropout": 0.2,
    "use_batchnorm": True,
    "lr_scheduler": "cosine",
    # CIFAR-10 normalization and augmentation (same for all cases).
    "calc_meanstd": True,
    "add_crop": True,
    "crop_padding": 4,
    "horizontal_flip_prob": 0.5,
    "add_jitter": True,
    "jitter_brightness": 0.2,
    "jitter_contrast": 0.2,
    "jitter_saturation": 0.2,
    "jitter_hue": 0.05,
    "add_random_erasing": False,
    "output_dir": "./Results",
    "wandb_project": "CS6886W_Assignment2_MobileNetV2",
    "wandb_group": "Q1_Controlled",
    "wandb_mode": "online",  # change to offline or disabled if needed
}


# TC01-TC12: All cases are compared against TC01 (the baseline).
# test_id identifies the altered parameter/value and also becomes the
# output folder name and W&B run name.
# Changes contain ONLY the settings that differ from the baseline.
TEST_CASES = [
    {"test_id": "TC01_baseline",               "category": "baseline",       "changes": {}},
    {"test_id": "TC02_width=0.5",              "category": "width",          "changes": {"width_mult": 0.5}},
    {"test_id": "TC03_width=0.75",             "category": "width",          "changes": {"width_mult": 0.75}},
    {"test_id": "TC04_dropout=0.0",            "category": "dropout",        "changes": {"dropout": 0.0}},
    {"test_id": "TC05_dropout=0.3",            "category": "dropout",        "changes": {"dropout": 0.3}},
    {"test_id": "TC06_batchnorm=OFF",          "category": "batchnorm",      "changes": {"use_batchnorm": False}},
    # Optimizer recipe: Adam requires a different appropriate starting LR.
    {"test_id": "TC07_optimizer=adam_lr=0.001", "category": "optimizer",       "changes": {"optimizer": "adam", "learning_rate": 0.001}},
    {"test_id": "TC08_lr=0.01",               "category": "learning_rate",  "changes": {"learning_rate": 0.01}},
    {"test_id": "TC09_scheduler=step",        "category": "scheduler",      "changes": {"lr_scheduler": "step"}},
    {"test_id": "TC10_scheduler=none",        "category": "scheduler",      "changes": {"lr_scheduler": "none"}},
    {"test_id": "TC11_weight_decay=0",        "category": "weight_decay",   "changes": {"weight_decay": 0.0}},
    {"test_id": "TC12_batch_size=64",         "category": "batch_size",     "changes": {"batch_size": 64}},
]


SUMMARY_FIELDS = [
    "test_id", "category", "width_mult", "dropout", "use_batchnorm",
    "optimizer", "learning_rate", "lr_scheduler", "weight_decay",
    "batch_size", "epochs", "best_epoch", "best_validation_accuracy",
    "trainable_parameters", "checkpoint", "test_loss", "test_accuracy",
    "test_top1_accuracy_percent", "test_correct", "test_total",
]


class Q1Run:
    """Trains and tests every TC, selecting best model by validation accuracy only."""

    def __init__(self, config):
        self.config = config

    def run(self):
        base = self.config
        root = Path(base.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        results = []
        print("=" * 88)
        print("QUESTION 1: MOBILENETV2 CONTROLLED HYPERPARAMETER EXPERIMENTS")
        print(f"Total planned test cases: {len(TEST_CASES)}")
        print("=" * 88)

        for number, case in enumerate(TEST_CASES, start=1):
            test_id = case["test_id"]
            settings = {**BASE_HYPERPARAMETERS, **case["changes"]}
            cfg = replace(base, **settings, mode="train", run_dir=None, run_name=test_id)
            cfg.validate()

            print("\n" + "#" * 88)
            print(f"TEST CASE {number:02d}/{len(TEST_CASES)}: {test_id}")
            print(f"CATEGORY: {case['category']}")
            print(f"OVERRIDES FROM BASELINE: {case['changes'] or 'None (baseline)'}")
            print("#" * 88)

            control = Control()
            control.setconfig(cfg)
            result = control.start()
            result["test_id"] = test_id
            result["category"] = case["category"]

            # Evaluate this TC's best-validation checkpoint on the test set.
            # Report test accuracy for all cases, but NEVER select by test accuracy.
            test_cfg = replace(
                cfg, mode="test", run_dir=str(Path(result["checkpoint"]).parent),
                run_name=test_id,
            )
            test_control = Control()
            test_control.setconfig(test_cfg)
            metrics = test_control.start()
            result.update({
                "test_loss": metrics["test/loss"],
                "test_accuracy": metrics["test/accuracy"],
                "test_top1_accuracy_percent": metrics["test/top1_accuracy_percent"],
                "test_correct": metrics["test/correct"],
                "test_total": metrics["test/total"],
            })
            run_dir = Path(result["checkpoint"]).parent
            (run_dir / "result.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            results.append(result)
            print(f"{test_id}: test top-1 accuracy = "
                  f"{result['test_top1_accuracy_percent']:.2f}% "
                  f"({result['test_correct']}/{result['test_total']})")

            # Recreate summary after each run: partial results survive interruptions.
            with (root / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
                writer.writeheader()
                for item in results:
                    config = item["configuration"]
                    writer.writerow({
                        "test_id": item["test_id"],
                        "category": item["category"],
                        **{k: config[k] for k in SUMMARY_FIELDS if k in config},
                        "best_epoch": item["best_epoch"],
                        "best_validation_accuracy": item["best_validation_accuracy"],
                        "trainable_parameters": item["trainable_parameters"],
                        "checkpoint": item["checkpoint"],
                        "test_loss": item["test_loss"],
                        "test_accuracy": item["test_accuracy"],
                        "test_top1_accuracy_percent": item["test_top1_accuracy_percent"],
                        "test_correct": item["test_correct"],
                        "test_total": item["test_total"],
                    })

        winner = max(results, key=lambda item: item["best_validation_accuracy"])
        (root / "best_run.json").write_text(json.dumps(winner, indent=2), encoding="utf-8")
        print(f"\nBEST VALIDATION TEST CASE: {winner['test_id']}")
        print(f"BEST VALIDATION ACCURACY: {winner['best_validation_accuracy'] * 100:.2f}%")

        # All 12 checkpoints have already been tested. Select using validation only.
        # The final summary simply reports the selected model's existing test result.
        final = {
            "selected_test_id": winner["test_id"],
            "selection_metric": "best_validation_accuracy",
            "best_validation_accuracy": winner["best_validation_accuracy"],
            "test/loss": winner["test_loss"],
            "test/accuracy": winner["test_accuracy"],
            "test/top1_accuracy_percent": winner["test_top1_accuracy_percent"],
            "test/correct": winner["test_correct"],
            "test/total": winner["test_total"],
            "checkpoint": winner["checkpoint"],
            "note": "Test results are descriptive; model was selected using validation accuracy.",
        }
        (root / "final_test_summary.json").write_text(
            json.dumps(final, indent=2), encoding="utf-8"
        )
        print(f"Completed {len(results)} training cases and {len(results)} test evaluations.")
        print(f"SELECTED MODEL TEST TOP-1: {winner['test_top1_accuracy_percent']:.2f}%")
        return final


def main():
    # No CLI parsing: the complete experiment specification is hardcoded above.
    config = ArgConfig(**BASE_HYPERPARAMETERS)
    config.validate()
    return Q1Run(config).run()


if __name__ == "__main__":
    main()
