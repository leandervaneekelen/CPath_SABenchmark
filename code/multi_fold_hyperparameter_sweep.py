import wandb
import argparse
import yaml
import numpy as np
import os
from functools import partial

from train_classification_mccv import main as train_classification
from train_survival_mccv import main as train_survival
from constants import DIMENSIONS_PER_EMBEDDER


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
        "--max_runs", type=int, default=1, help="number of random hyperparameter runs"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="",
        choices=[
            "AB-MIL",
            "AB-MIL_FC_small",
            "AB-MIL_FC_big",
            "CLAM_SB",
            "CLAM_MB",
            "transMIL",
            "DS-MIL",
            "VarMIL",
            "GTP",
            "PatchGCN",
            "DeepGraphConv",
            "ViT_MIL",
            "DTMIL",
            "LongNet_ViT",
        ],
        help="which aggregation method to use",
    )
    parser.add_argument("--data", type=str, default="", help="Path to dataset in excel")
    parser.add_argument(
        "--y_label",
        type=str,
        default="y_1year",
        help="Column name of y_labels in dataset sheet",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="",
        choices=DIMENSIONS_PER_EMBEDDER.keys(),
        help="which encoder to use",
    )
    parser.add_argument(
        "--nepochs", type=int, default=40, help="number of epochs (default: 40)"
    )
    parser.add_argument("--random_seed", default=0, type=int, help="random seed")

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

    # Start new wandb sweep run to save the aggregated metrics to
    sweep_run = wandb.init()
    sweep_id = sweep_run.sweep_id or "unknown"
    sweep_url = sweep_run.get_sweep_url()
    project_url = sweep_run.get_project_url()
    sweep_group_url = f'{project_url}/groups/{sweep_id}'
    sweep_run.notes = sweep_group_url
    sweep_run.save()
    sweep_run_name = sweep_run.name or sweep_run.id or "unknown_2"
    sweep_run_id = sweep_run.id
    sweep_run.finish()
    wandb.sdk.wandb_setup._setup(_reset=True)

    # Set task
    train = train_classification if args.task == "classification" else train_survival

    # Do training run for each fold
    # folds = [1, 2, 3, 4, 5]
    folds = [1, 2] # Debugging purposes, remove later
    evaluation_metrics = []
    for fold in folds:

        # Give wandb identifiers to each run
        reset_wandb_env()
        setattr(args, "sweep_id", sweep_id)
        setattr(args, "sweep_run_name", sweep_run_name)
        setattr(args, "sweep_run_id", sweep_run_id)

        # Sync hyperparameters
        for key, value in sweep_run.config.items():
            setattr(args, key, value)

        # Train on fold
        setattr(args, "mccv", fold)
        eval_metric = train(args)
        evaluation_metrics.append(eval_metric)

    # Resume sweep run and log the average evaluation metric
    sweep_run = wandb.init(id=sweep_run_id, resume="must")
    avg_eval_metric = np.mean(evaluation_metrics)
    sweep_run.log({"avg_eval_metric": avg_eval_metric})
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