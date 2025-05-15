import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import argparse
import random
import yaml
import wandb

from sklearn.metrics import roc_auc_score
from pathlib import Path

import datasets
import modules
from constants import DIMENSIONS_PER_EMBEDDER

# from aggregator import NestedTensor

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
    "--mccv",
    default=1,
    type=int,
    choices=list(range(1, 22)),
    help="which seed (default: 1/20)",
)
parser.add_argument(
    "--ndim", default=512, type=int, help="output dimension of feature extractor"
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
    "--warmup_epochs",
    default=10,
    type=int,
    help="Number of epochs for the linear learning-rate warm up.",
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
parser.add_argument("--random_seed", default=0, type=int, help="random seed")

# Weight and Bias Config
parser.add_argument("--wandb_project", type=str, help="name of project in wandb")
parser.add_argument("--wandb_note", type=str, help="note of project in wandb")
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


def main(config=None):
    # Initialize wandb
    wandb.init(project=args.wandb_project, notes=args.wandb_note)

    if args.sweep_config:
        for key, value in wandb.config.items():
            if hasattr(args, key):
                setattr(args, key, value)

        args_dict = vars(args) if isinstance(args, argparse.Namespace) else args
        for key, value in args_dict.items():
            if key not in wandb.config:
                wandb.config[key] = value

        wandb.config.update(args_dict, allow_val_change=True)

    # Randomness
    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    # Set datasets
    train_dset, val_dset, test_dset = datasets.get_classification_datasets(
        mccv=args.mccv,
        data=args.data,
        y_label=args.y_label,
        encoder=args.encoder,
        method=args.method,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=lambda worker_id: np.random.seed(args.random_seed + worker_id),
        generator=torch.Generator().manual_seed(args.random_seed),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dset, batch_size=1, shuffle=False, num_workers=args.workers
    )
    test_loader = (
        torch.utils.data.DataLoader(
            test_dset, batch_size=1, shuffle=False, num_workers=args.workers
        )
        if test_dset is not None
        else None
    )

    # Get model
    model = modules.get_aggregator(
        method=args.method, ndim=args.ndim, n_classes=2
    )  # TODO: make `n_classes` dynamic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set loss
    criterion = nn.CrossEntropyLoss().to(device)

    # Set optimizer
    params_groups = get_params_groups(model)
    optimizer = optim.AdamW(params_groups)

    # Set schedulers
    lr_schedule = cosine_scheduler(
        args.lr,
        args.lr_end,
        args.nepochs,
        len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.nepochs,
        len(train_loader),
    )
    cudnn.benchmark = True

    best_auc = 0.0
    # Main training loop
    for epoch in range(args.nepochs + 1):

        if epoch == 0:  # Special case for testing feature extractor
            # Validation logic for feature extractor testing
            probs, val_loss = test(epoch, val_loader, model, criterion)
            val_auc = roc_auc_score(val_loader.dataset.df.y, probs)
            # Log this epoch with a note
            wandb.log({"epoch": epoch, "val_auc": val_auc, "val_loss": val_loss})
        else:
            # Regular training and validation logic
            train_loss = train(
                epoch,
                train_loader,
                model,
                criterion,
                optimizer,
                lr_schedule,
                wd_schedule,
            )
            # Get the current learning rate from the first parameter group
            current_lr = optimizer.param_groups[0]["lr"]
            current_wd = optimizer.param_groups[0]["weight_decay"]
            probs, val_loss = test(epoch, val_loader, model, criterion)
            val_auc = roc_auc_score(val_loader.dataset.df.y, probs)
            # Regular AUC logging
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "lr_step": current_lr,
                    "wd_step": current_wd,
                }
            )

            # Check if the current model is the best one
            if val_auc > best_auc:
                print(f"New best model found at epoch {epoch} with AUC: {val_auc}")
                best_auc = val_auc
                # Log this event to wandb
                wandb.run.summary["best_auc"] = val_auc
                wandb.run.summary["best_epoch"] = epoch

    if args.data in ["camelyon16"] and test_loader != None:
        probs, _ = test(epoch, test_loader, model, criterion)
        test_auc = roc_auc_score(test_loader.dataset.df.y, probs)
        # Log this epoch with a note
        wandb.log({"epoch": epoch, "test_auc": test_auc})

    # Model saving logic
    if epoch == args.nepochs:  # only save the last model to artifact
        model_filename = Path(args.output_dir) / f"{args.output_name}.pt"
        obj = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "auc": val_auc,
            "optimizer": optimizer.state_dict(),
        }
        torch.save(obj, model_filename)
        print(f"Saved final model at epoch {epoch}")

    # # Create a wandb Artifact and add the file to it
    # model_artifact = wandb.Artifact('final_model_checkpoint', type='model')
    # model_artifact.add_file(model_filename)
    # wandb.log_artifact(model_artifact)

    wandb.finish()


def test(epoch, loader, model, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set model in test mode
    model.eval()
    # Initialize loss
    running_loss = 0.0
    # Initialize probability vector
    probs = torch.FloatTensor(len(loader)).to(device)
    # Loop through batches
    with torch.no_grad():
        for i, batch in enumerate(loader):

            # Get features, reshape and copy to GPU
            if args.method in ["ViT_MIL", "DTMIL"]:
                feat = batch["feat_map"].float().permute(0, 3, 1, 2)
            else:
                feat = batch["features"].squeeze(0)
            feat = feat.to(device)

            # Forward pass
            if args.method in ["GTP"]:
                adj = batch["adj_mtx"].float().to(device)
                mask = batch["mask"].float().to(device)
                results_dict = model(feat, adj, mask)
            elif args.method in ["PatchGCN", "DeepGraphConv"]:
                edge_index = batch["edge_index"].squeeze(0).to(device)
                edge_latent = batch["edge_latent"].squeeze(0).to(device)
                results_dict = model(
                    feat=feat, edge_index=edge_index, edge_latent=edge_latent
                )
            # elif args.method in ['DTMIL']:
            #     mask = input['mask'].bool().to(device)
            #     tensors = NestedTensor(feat, mask)
            #     results_dict = model(tensors)
            else:
                results_dict = model(feat)

            logits, Y_prob, Y_hat = (
                results_dict[key] for key in ["logits", "Y_prob", "Y_hat"]
            )
            ## Calculate loss
            target = batch["target"].long().to(device)
            loss = criterion(logits, target)
            if args.method in ["GTP"]:
                mc1 = results_dict["mc1"]
                o1 = results_dict["o1"]
                loss = loss + mc1 + o1
            running_loss += loss.item()

            ## Clone output to output vector
            probs[i] = Y_prob.detach()[:, 1].item()
    mean_val_loss = running_loss / len(loader)
    return mean_val_loss, probs.cpu().numpy()


def train(epoch, loader, model, criterion, optimizer, lr_schedule, wd_schedule):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set model in training mode
    model.train()
    # Initialize loss
    running_loss = 0.0
    # Loop through batches
    for i, batch in enumerate(loader):
        ## Update weight decay and learning rate according to their schedule
        it = len(loader) * (epoch - 1) + i  # global training iteration
        for j, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if j == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        target = batch["target"].long().to(device)

        # Get features, reshape and copy to GPU
        if args.method in ["ViT_MIL", "DTMIL"]:
            feat = batch["feat_map"].float().permute(0, 3, 1, 2)
        else:
            feat = batch["features"].squeeze(0)
        feat = feat.to(device)

        # Forward pass
        if args.method == "GTP":
            adj = batch["adj_mtx"].float().to(device)
            mask = batch["mask"].float().to(device)
            results_dict = model(feat, adj, mask)
            logits = results_dict["logits"]
            mc1 = results_dict["mc1"]
            o1 = results_dict["o1"]
            ## Calculate loss
            loss = criterion(logits, target)
            loss = loss + mc1 + o1

        elif args.method in ["PatchGCN", "DeepGraphConv"]:
            edge_index = batch["edge_index"].squeeze(0).to(device)
            edge_latent = batch["edge_latent"].squeeze(0).to(device)
            results_dict = model(
                feat=feat, edge_index=edge_index, edge_latent=edge_latent
            )
            logits = results_dict["logits"]
            ## Calculate loss
            loss = criterion(logits, target)

        # elif args.method in ['DTMIL']:
        #     mask = input['mask'].bool().to(device)
        #     tensors = NestedTensor(feat, mask)
        #     results_dict = model(tensors)

        else:
            results_dict = model(feat)
            logits = results_dict["logits"]
            ## Calculate loss
            loss = criterion(logits, target)

        running_loss += loss.item()

        # Optimization step with optional gradient accumulation
        loss /= args.gradient_accumulation_steps
        loss.backward()
        if (i + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    mean_train_loss = running_loss / len(loader)
    return mean_train_loss


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
    metric="val_auc",
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
        metric="val_auc",
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

    # Dim of features
    if args.encoder in DIMENSIONS_PER_EMBEDDER:
        args.ndim = DIMENSIONS_PER_EMBEDDER[args.encoder]

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

        wandb.agent(sweep_id, function=lambda: main())

    else:
        print("args:", args)
        main(args)
