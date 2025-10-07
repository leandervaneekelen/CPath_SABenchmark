import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim

import pandas as pd
import numpy as np
import argparse
import random
import yaml
import math
import wandb

from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from pathlib import Path
from functools import partial

# from aggregator import NestedTensor
import datasets
import modules
from constants import DIMENSIONS_PER_EMBEDDER
from survival_losses import LossFactory

parser = argparse.ArgumentParser()

# I/O PARAMS
parser.add_argument("--output_dir", type=str, default=".", help="output directory")
parser.add_argument(
    "--output_name",
    type=str,
    default="final_model",
    help="name of .pt file saved at end of run",
)
parser.add_argument(
    "--log", type=str, default="convergence.csv", help="name of log file"
)
parser.add_argument(
    "--method",
    type=str,
    default="",
    choices=modules.AGGREGATION_METHODS,
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
    "--tile_index", type=str, default=None, help="tile index column name"
)

parser.add_argument(
    "--fold",
    default=1,
    type=str,
    help="which fold to use; column name in data csv",
)


# OPTIMIZATION PARAMS
parser.add_argument(
    "--momentum", default=0.9, type=float, help="momentum (default: 0.9)"
)
parser.add_argument(
    "--batch_size", default=32, type=int, help="batch size (default: 32)"
)
parser.add_argument(
    "--gradient_accumulation_steps",
    default=1,
    type=int,
    help="Number of batches to accumulate loss over before doing an optimizer step.",
)
parser.add_argument(
    "--lr",
    default=0.0005,
    type=float,
    help="""Learning rate at the end of linear warmup (highest LR used during training). The learning rate is linearly scaled with the batch size, and specified here for a reference batch size of 256.""",
)
parser.add_argument(
    "--lr_end",
    type=float,
    default=1e-6,
    help="""Target LR at the end of optimization. We use a cosine LR schedule with linear warmup.""",
)
parser.add_argument(
    "--loss",
    type=str,
    default="nll",
    choices=["nll", "cox"],
    help="""Loss function to use. 'nll' is the negative log-likelihood loss, and 'cox' is the Cox proportional hazards loss.""",
)
parser.add_argument(
    "--warmup_epochs",
    default=10,
    type=int,
    help="Number of epochs for the linear learning-rate warm up.",
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0.0,
    help="Dropout rate (default: 0.0)",
)
parser.add_argument(
    "--hidden_dim",
    type=int,
    default=512,
    help="Hidden dimension for models that use it (default: 512)",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.04,
    help="""Initial value of the weight decay. With ViT, a smaller value at the beginning of training works well.""",
)
parser.add_argument(
    "--weight_decay_end",
    type=float,
    default=0.4,
    help="""Final value of the weight decay. We use a cosine schedule for WD and using a larger decay by the end of training improves performance for ViTs.""",
)
parser.add_argument(
    "--nepochs", type=int, default=40, help="number of epochs (default: 40)"
)
parser.add_argument(
    "--workers",
    default=10,
    type=int,
    help="number of data loading workers (default: 10)",
)
parser.add_argument("--random_seed", default=None, type=int, help="random seed")


# Balanced sampling
parser.add_argument(
    "--balanced_sampling",
    default=True,
    action="store_true",
    help="Use balanced sampling to handle class imbalance",
)

# Tile subsampling augmentation
parser.add_argument(
    "--n_subsamples",
    type=int,
    default=None,
    help="Number of tiles to randomly subsample from each slide for augmentation (None = use all tiles)",
)

# Gaussian noise augmentation
parser.add_argument(
    "--noise_std",
    type=float,
    default=None,
    help="Standard deviation for Gaussian noise augmentation on embeddings (None = no noise)",
)

# Memory caching
parser.add_argument(
    "--cache_in_memory",
    action="store_true",
    help="Cache all features in memory at initialization to speed up training",
)

# Weight and Bias Config
parser.add_argument(
    "--wandb_project", type=str, default=None, help="name of project in wandb"
)
parser.add_argument(
    "--wandb_note", type=str, default=None, help="note of project in wandb"
)
parser.add_argument(
    "--sweep_config", type=str, help="Path to the sweep configuration YAML file"
)
parser.add_argument(
    "--sweep_name",
    type=str,
    default=None,
    help="optional override for  wandb sweep name",
)
parser.add_argument(
    "--parameter_path", type=str, help="Read hyperparameters after tuning"
)


def set_random_seed(seed_value):
    random.seed(seed_value)  # Python random module.
    np.random.seed(seed_value)  # Numpy module.
    torch.manual_seed(
        seed_value
    )  # Sets the seed for generating random numbers for CPU.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(
            seed_value
        )  # Sets the seed for generating random numbers on all GPUs.
        torch.cuda.manual_seed_all(
            seed_value
        )  # Sets the seed for generating random numbers on all GPUs.
        torch.backends.cudnn.deterministic = (
            True  # Makes CUDA operations deterministic.
        )
        torch.backends.cudnn.benchmark = False  # If True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.


def main(config):

    # Initialize wandb
    is_main_process = __name__ == "__main__"
    if is_main_process:
        # Regular standalone run
        run = wandb.init(
            project=config.wandb_project, notes=config.wandb_note, config=config
        )
    else:
        # This is a fold run within a sweep - log values with distinct names
        logging_prefix = f"fold_{config.fold}/"
        run = wandb.run
        run_name = f"{run.name}_{config.fold}"
        setattr(config, "output_name", run_name)

        # In case of hyperparameter tuning: do two-way sync between wandb config and local config
        # (to facilitate hyperparameter tuning via external config files)
        for key, value in run.config.items():
            if key not in ["fold", "random_seed"]:
                setattr(config, key, value)

        arg_dict = vars(config) if isinstance(config, argparse.Namespace) else config
        to_update = {}
        for key, value in arg_dict.items():
            if key not in run.config:
                to_update[key] = value
        run.config.update(to_update, allow_val_change=True)

    # Randomness
    if config.random_seed is not None:
        set_random_seed(config.random_seed)

    if not Path(config.output_dir).exists():
        Path(config.output_dir).mkdir(parents=True)

    # Set datasets
    train_dset, val_dset, test_dset = datasets.get_survival_datasets(
        fold=config.fold,
        data=config.data,
        y_label=config.y_label,
        encoder=config.encoder,
        method=config.method,
        tile_index=config.tile_index,
        n_subsamples=config.n_subsamples,
        random_seed=config.random_seed,
        noise_std=config.noise_std,
        cache_in_memory=config.cache_in_memory,
    )

    # Create balanced sampler if specified
    train_sampler = None
    shuffle_train = True
    if config.balanced_sampling:
        train_sampler = datasets.create_balanced_sampler(train_dset)
        shuffle_train = False  # Sampler shuffles data, so disable shuffle in DataLoader

    n_bins = 1 if config.loss == "cox" else train_dset.n_bins
    collate_fn = datasets.collate_batch if config.batch_size > 1 else None
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=config.batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=config.workers,
        worker_init_fn=lambda worker_id: np.random.seed(config.random_seed + worker_id),
        generator=torch.Generator().manual_seed(config.random_seed),
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=config.workers,
    )
    test_loader = (
        torch.utils.data.DataLoader(
            test_dset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            collate_fn=collate_fn,
        )
        if test_dset is not None
        else None
    )

    # Get model
    ndim = DIMENSIONS_PER_EMBEDDER[config.encoder]
    model = modules.get_aggregator(
        method=config.method,
        n_classes=n_bins,
        ndim=ndim,
        dropout=config.dropout if hasattr(config, "dropout") else 0.0,
        hidden_dim=config.hidden_dim if hasattr(config, "hidden_dim") else 512,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set loss
    criterion = LossFactory(loss=config.loss, alpha=0.4).get_loss()

    # Set optimizer
    params_groups = get_params_groups(model)
    optimizer = optim.AdamW(params_groups)
    # optimizer = optim.SGD(params_groups, momentum=config.momentum)

    # Set schedulers
    lr_schedule = cosine_scheduler(
        config.lr,
        config.lr_end,
        config.nepochs,
        len(train_loader),
        warmup_epochs=config.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        config.weight_decay,
        config.weight_decay_end,
        config.nepochs,
        len(train_loader),
    )
    cudnn.benchmark = True

    best_c_index = 0.0
    # Main training loop
    for epoch in range(config.nepochs + 1):

        if epoch == 0:  # Special case for testing feature extractor
            # Validation logic for feature extractor testing
            val_loss, val_c_index, _ = test(val_loader, model, criterion)
            # Log this epoch with a note
            run.log(
                {
                    "epoch": epoch,
                    f"{logging_prefix}val_c_index": val_c_index,
                    f"{logging_prefix}val_loss": val_loss,
                },
            )
        else:
            # Regular training and validation logic
            train_loss, train_c_index = train(
                epoch,
                train_loader,
                model,
                criterion,
                optimizer,
                lr_schedule,
                wd_schedule,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
            )
            val_loss, val_c_index, val_risk_scores = test(val_loader, model, criterion)

            # _, mean_auc, _ = get_cumulative_dynamic_auc(
            #     train_dset.df, val_dset.df, val_risk_scores, config.y_label
            # )

            # Regular logging
            current_lr = optimizer.param_groups[0]["lr"]
            current_wd = optimizer.param_groups[0]["weight_decay"]
            run.log(
                {
                    "epoch": epoch,
                    f"{logging_prefix}train_loss": train_loss,
                    f"{logging_prefix}val_loss": val_loss,
                    f"{logging_prefix}train_c_index": train_c_index,
                    f"{logging_prefix}val_c_index": val_c_index,
                    f"{logging_prefix}val_mean_cumulative_auc": mean_auc,
                    f"{logging_prefix}lr_step": current_lr,
                    f"{logging_prefix}wd_step": current_wd,
                },
            )

            # Check if the current model is the best one
            if val_c_index > best_c_index:
                print(
                    f"New best model found at epoch {epoch} with (validation) C-index: {val_c_index}"
                )
                best_c_index = val_c_index
                run.summary[f"{logging_prefix}best_c_index"] = val_c_index
                model_filename = (
                    Path(config.output_dir) / f"{config.output_name}_best_cindex.pt"
                )
                obj = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "c_index": val_c_index,
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(obj, model_filename)

    if config.data in ["camelyon16"] and test_loader != None:
        _, test_c_index = test(test_loader, model, criterion)
        # Log this epoch with a note
        wandb.log(
            {"epoch": epoch, f"{logging_prefix}test_c_index": test_c_index},
        )

    # Model saving logic
    if epoch == config.nepochs:  # only save the last model to artifact
        model_filename = Path(config.output_dir) / f"{config.output_name}.pt"
        obj = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "c_index": val_c_index,
            "optimizer": optimizer.state_dict(),
        }
        torch.save(obj, model_filename)
        print(f"Saved final model at epoch {epoch}")

    if is_main_process:
        run.finish()
    return val_c_index


def test(loader, model, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    running_loss = 0.0
    censoring, times_to_events, risk_scores = [], [], []

    # Loop through batches
    with torch.no_grad():
        for i, batch in enumerate(loader):

            # Get targets
            discrete_labels, time_to_events, censored = (
                batch["target"].to(device),
                batch["time_to_event"].float(),
                batch["censored"].bool().to(device),
            )

            # Forward pass
            results_dict = modules.model_forward_pass(
                batch, model, model.method, device
            )

            # Calculate discrete hazards/survival function
            logits = results_dict["logits"]  # [batch_size, n_bins]
            hazards = torch.sigmoid(logits)
            surv = torch.cumprod(1 - hazards, dim=1)  # [batch_size]

            ## Calculate loss
            mc1 = results_dict.get("mc1", 0.0)  # Optional aux loss from GTP
            o1 = results_dict.get("o1", 0.0)  # Optional aux loss from GTP
            loss = (
                criterion(
                    hazards=hazards,
                    survival=surv,
                    Y=discrete_labels,
                    c=censored,
                    alpha=0.0,
                )
                + mc1
                + o1
            )
            running_loss += loss.item()

            # Calculate risk scores and collect outputs
            risk = -torch.sum(surv, dim=1)  # [batch_size]
            risk_scores.extend(risk.detach().clone().tolist())
            censoring.extend(censored.detach().clone().tolist())
            times_to_events.extend(time_to_events.clone().tolist())

    # Calculate c-index of the epoch
    c_index = concordance_index_censored(
        [bool(1 - c) for c in censoring], times_to_events, risk_scores
    )[0]

    # Return metrics & risk scores
    mean_val_loss = running_loss / len(loader)
    return mean_val_loss, c_index, risk_scores


def train(
    epoch,
    loader,
    model,
    criterion,
    optimizer,
    lr_schedule,
    wd_schedule,
    gradient_accumulation_steps=1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    running_loss = 0.0
    censoring, times_to_events = [], []
    risk_scores, labels = [], []

    # Loop through batches
    for i, batch in enumerate(loader):
        ## Update weight decay and learning rate according to their schedule
        it = len(loader) * (epoch - 1) + i  # global training iteration
        for j, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if j == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # Get targets
        discrete_labels, time_to_events, censored = (
            batch["target"].to(device),
            batch["time_to_event"].float(),
            batch["censored"].bool().to(device),
        )
        # Forward pass
        results_dict = modules.model_forward_pass(batch, model, model.method, device)

        # Calculate discrete hazards/survival function
        logits = results_dict["logits"]  # [batch_size, n_bins]
        hazards = torch.sigmoid(logits)
        surv = torch.cumprod(1 - hazards, dim=1)  # [batch_size]

        # Calculate loss
        mc1 = results_dict.get("mc1", 0.0)  # Optional aux loss from GTP
        o1 = results_dict.get("o1", 0.0)  # Optional aux loss from GTP
        loss = (
            criterion(hazards=hazards, survival=surv, Y=discrete_labels, c=censored)
            + mc1
            + o1
        )
        running_loss += loss.item()

        # Calculate risk scores and collect outputs
        risk = -torch.sum(surv, dim=1)  # [batch_size]

        # Collect outputs (handle both single and batch cases)
        risk_scores.extend(risk.detach().clone().tolist())
        labels.extend(discrete_labels.detach().clone().tolist())
        censoring.extend(censored.detach().clone().tolist())
        times_to_events.extend(time_to_events.clone().tolist())

        # Optimization step with optional gradient accumulation
        loss /= gradient_accumulation_steps
        loss.backward()
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Calculate c-index of the epoch
    c_index = concordance_index_censored(
        [bool(1 - c) for c in censoring], times_to_events, risk_scores
    )[0]

    mean_train_loss = running_loss / len(loader)
    return mean_train_loss, c_index


def get_cumulative_dynamic_auc(
    train_data, test_data, test_risk_scores, times, verbose=False
):
    cols = ["censored", "time_to_event"]
    train_tuples = train_data[cols].values
    tune_tuples = test_data[cols].values
    survival_train = np.array(
        list(zip(train_tuples[:, 0], train_tuples[:, 1])), dtype=np.dtype("bool,float")
    )
    survival_tune = np.array(
        list(zip(tune_tuples[:, 0], tune_tuples[:, 1])), dtype=np.dtype("bool,float")
    )

    train_min, train_max = (
        train_data["time_to_event"].min(),
        train_data["time_to_event"].max(),
    )
    test_min, test_max = (
        test_data["time_to_event"].min(),
        test_data["time_to_event"].max(),
    )
    min_y = math.ceil(test_min / 12)  # Convert months to years
    max_y = math.floor(test_max / 12)
    times = np.arange(min_y, max_y, 1)  # Evaluate AUC at 1-year time intervals
    if train_min <= test_min < test_max < train_max:
        auc, mean_auc = cumulative_dynamic_auc(
            survival_train, survival_tune, test_risk_scores, times * 12
        )
    else:
        if verbose:
            print(
                f"test data ({test_min},{test_max}) is not within time range of training data ({train_min},{train_max})"
            )
        auc, mean_auc = None, None
    return auc, mean_auc, times


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):

    assert (
        warmup_epochs < epochs
    ), f"Warmup epochs ({warmup_epochs}) must be less than total epochs ({epochs})."

    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


# Define the function to read best hyperparameters
def find_best_hyperparameters(
    project_path,
    data_filter,
    label_filter,
    encoder_filter,
    method_filter,
    metric="val_c_index",
    config_interest=None,
):
    # Initialize the API and get the runs
    api = wandb.Api()
    runs = api.runs(project_path)

    # Create a list of dictionaries for each run with config and the metric
    runs_data = []
    for run in runs:
        # Flatten the config and include the metric of interest
        run_data = {f"config.{k}": v for k, v in run.config.items()}
        run_data[metric] = run.summary.get(metric)
        runs_data.append(run_data)

    # Create a DataFrame from the list of dictionaries
    runs_df = pd.DataFrame(runs_data)

    # Apply filters to the DataFrame
    filtered_df = runs_df[
        (runs_df["config.data"] == data_filter)
        & (runs_df["config.y_label"] == label_filter)
        & (runs_df["config.encoder"] == encoder_filter)
        & (runs_df["config.method"] == method_filter)
    ]

    # Check if the filtered DataFrame is empty
    if filtered_df.empty:
        raise ValueError("No runs found with the specified filters.")

    # Sort by the specified metric to find the best run
    best_run = filtered_df.sort_values(by=metric, ascending=False).iloc[0]

    # Extract the config of the best run
    # Check if the key starts with 'config.' and is in the list of interest
    if config_interest:
        best_config = {
            key.split("config.")[1]: value
            for key, value in best_run.items()
            if key.startswith("config.") and key.split("config.")[1] in config_interest
        }
    else:
        best_config = {
            key.split("config.")[1]: value
            for key, value in best_run.items()
            if key.startswith("config.")
        }

    return best_config


# Function to update the config with best hyperparameters
def update_args_with_best_hyperparameters(args):
    print(f"Reading fine-tuned hyperparameters from wandb {args.parameter_path}")
    best_hyperparameters = find_best_hyperparameters(
        args.parameter_path,
        args.data,
        args.y_label,
        args.encoder,
        args.method,
        metric="val_c_index",
        config_interest=["lr", "weight_decay", "momentum"],
    )

    # Update the args namespace with the best hyperparameters
    for param, value in best_hyperparameters.items():
        setattr(args, param, value)

    print(
        f"Hyperparameters {best_hyperparameters} from wandb {args.parameter_path} updated!"
    )


# Function to update the config with selected hyperparameters
def update_args_with_selected_hyperparameters(args):
    print(f"Reading selected hyperparameters from {args.parameter_path}")

    # Load hyperparameters from CSV
    df_hyperparameters = pd.read_csv(args.parameter_path)

    # Selecting hyperparameters based on specific criteria
    selected_row = df_hyperparameters[
        (df_hyperparameters["data"] == args.data)
        & (df_hyperparameters["y_label"] == args.y_label)
        & (df_hyperparameters["encoder"] == args.encoder)
        & (df_hyperparameters["method"] == args.method)
    ].iloc[0]

    # Update the args namespace with the selected hyperparameters
    for param in [
        "lr",
        "weight_decay",
        "data",
        "encoder",
        "method",
    ]:  # Add other hyperparameters as needed
        if param in selected_row:
            setattr(args, param, selected_row[param])

    print(
        f"Hyperparameters {selected_row[['lr', 'weight_decay']]} from {args.parameter_path} updated!"
    )


if __name__ == "__main__":
    args = parser.parse_args()

    # Update args with hyperparameters based on the file extension of parameter_path
    if args.parameter_path:
        if args.parameter_path.endswith(".csv"):
            update_args_with_selected_hyperparameters(args)
        else:
            update_args_with_best_hyperparameters(args)

    if (
        args.sweep_config
    ):  # sweep parameters for hyperparameter tuning or cross validation
        print(f"Find sweep configuration files at {args.sweep_config}")
        # Load sweep configuration from the YAML file
        with open(args.sweep_config, "r") as file:
            sweep_config = yaml.safe_load(file)
            if args.sweep_name:  # Optional override
                sweep_config["name"] = args.sweep_name
            parameter_names = list(sweep_config["parameters"].keys())
            print(f"parameter_names:{parameter_names}")

        # Initialize the sweep
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)

        wandb.agent(sweep_id, function=partial(main, args))

    else:
        print("args:", args)
        main(args)
