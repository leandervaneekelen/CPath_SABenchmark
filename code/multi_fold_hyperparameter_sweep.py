"""
All hyperparameter domains should be defined in the sweep_config.yaml file (e.g. 'batch_size' or 'encoder'), even if they are not part of the sweep and remain constant.
"""

import wandb
import argparse
import yaml
import numpy as np
import os
from functools import partial

from train_classification import main as train_classification
from train_survival import main as train_survival


def parse_args():
    parser = argparse.ArgumentParser()

    # I/O PARMAS
    parser.add_argument("--output_dir", type=str, default=".", help="output directory")
    parser.add_argument(
        "--workers",
        default=10,
        type=int,
        help="number of data loading workers (default: 10)",
    )

    # METHOD PARAMS
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        help="task type",
        choices=["classification", "survival"],
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=None,
        help="number of random hyperparameter runs",
    )
    parser.add_argument("--data", type=str, default="", help="Path to dataset in excel")
    parser.add_argument(
        "--y_label",
        type=str,
        default="y_1year",
        help="Column name of y_labels in dataset sheet",
    )

    # Weight and Bias Config
    parser.add_argument("--wandb_project", type=str, help="name of project in wandb")
    parser.add_argument("--wandb_note", type=str, help="note of project in wandb")
    parser.add_argument(
        "--sweep_config", type=str, help="Path to the sweep configuration YAML file"
    )

    return parser.parse_args()


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for key in os.environ.keys():
        if key.startswith("WANDB_") and key not in exclude:
            del os.environ[key]


def multi_fold_train(args):

    # Start new wandb sweep
    sweep_run = wandb.init()

    # Set task
    train = train_classification if args.task == "classification" else train_survival

    folds = ["cv1", "cv2", "cv3", "cv4"]
    evaluation_metrics = []

    for i, fold in enumerate(folds):

        # Sync hyperparameters from sweep
        for key, value in sweep_run.config.items():
            setattr(args, key, value)

        # Train on fold
        setattr(args, "fold", fold)
        setattr(args, "random_seed", i)
        eval_metric = train(args)
        evaluation_metrics.append(eval_metric)

    # Calculate and log the average evaluation metric
    for fold, metric in zip(folds, evaluation_metrics):
        sweep_run.summary[f"fold_{fold}_eval_metric"] = metric
    avg_eval_metric = np.mean(evaluation_metrics)
    sweep_run.summary["avg_eval_metric"] = avg_eval_metric

    sweep_run.finish()


def main(args):
    with open(args.sweep_config, "r") as f:
        sweep_config = yaml.safe_load(f)

    # Needed for proper error handling
    func = partial(multi_fold_train, args)

    def run_with_exception_logging():
        import sys
        import traceback

        try:
            func()
        except Exception as e:
            print(traceback.print_exc(), file=sys.stderr)

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(
        sweep_id,
        function=run_with_exception_logging,
        count=args.max_runs,
        project=args.wandb_project,
    )
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
