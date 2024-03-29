from src.models.mil_model import MILModel
from src.models.senet import *
from src.models.attentionDeepMil import *
from src.trainer import BaseTrainer
import argparse
import os
import yaml
from pathlib import Path

MODELS_DICT = { 
    'mil_model': MILModel,  
    'senet': SENet,
    'attentionDeepMil': Attention,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="mil_model",
        help="path to the config file",
    )

    parser.add_argument(
        "--checkpoint_name",
        type=str,
        help="name of the checkpoint to load",
    )

    parser.add_argument(
        "--device",
        type=str,
        help="device to use for training",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="float16",
        help="precision to use for training",
    )
    
    parser.add_argument(
        "--train_dir",
        type=str,
        default="/kaggle/input/dlmi-lymphocytosis/dlmi-lymphocytosis-classification/trainset",
        help="path to the training directory",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="/kaggle/input/dlmi-lymphocytosis/dlmi-lymphocytosis-classification/testset",
        help="path to the test directory",
    )
    parser.add_argument(
        "--df_train_path",
        type=str,
        default="/kaggle/input/dlmi-lymphocytosis/dlmi-lymphocytosis-classification/trainset/trainset_true.csv",
        help="path to the training dataframe",
    )
    parser.add_argument(
        "--df_test_path",
        type=str,
        default="/kaggle/input/dlmi-lymphocytosis/dlmi-lymphocytosis-classification/testset/testset_data.csv",
        help="path to the test dataframe",
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed")

    parser.add_argument("--epochs", type=int, help="number of epochs to train")
    parser.add_argument("--batch_size", type=int, help="batch size to use for training")
    parser.add_argument("--lr", type=float, help="learning rate to use for training")
    parser.add_argument(
        "--ft", action="store_true", help="whether to finetune the checkpoint if given"
    )

    parser.add_argument(
        "--is_debug",
        action="store_true",
        help="whether to run in debug mode",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="DLMI project",
        help="name of the wandb project for logs",
    )

    # TODO Parse the arguments to override the config when specified

    args = parser.parse_args()
    # Delete None arguments
    delattrs = []
    for arg in vars(args):
        if getattr(args, arg) is None:
            delattrs.append(arg)
    for arg in delattrs:
        delattr(args, arg)

    config_type = args.config.split("-")
    config_file = config_type[0]

    config_path = Path("configs") / (config_file + ".yaml")
    with open(config_path, "r") as f:
        config_read = yaml.safe_load(f)
    model = MODELS_DICT[config_file]

    config = config_read["default"]

    if len(config_type) > 1:
        config_model = config_type[1]
        # Careful here if parent is node in dict
        for arg in config_read[config_model]:
            config[arg].update(config_read[config_model][arg])

    config["name_model"] = config_file
    config["model_object"] = model
    config["config_name"] = args.config

    for arg in vars(args):
        config[arg] = getattr(args, arg)

    # Update epochs:
    if "epochs" in config:
        config["optim"]["max_epochs"] = config["epochs"]
        del config["epochs"]
    if "batch_size" in config:
        config["optim"]["batch_size"] = config["batch_size"]
        del config["batch_size"]
        config["optim"]["eval_batch_size"] = config["optim"]["batch_size"]
    if "lr" in config:
        config["optim"]["lr_initial"] = config["lr"]
        del config["lr"]

    if "checkpoint_name" in config:
        if config["ft"]:
            # TODO: Save the config of the checkpoint?
            checkpoint_dir = Path("runs/checkpoints/")
            configs_in_path = [
                x
                for x in checkpoint_dir.iterdir()
                if (
                    (".yaml" in x.name)
                    and (
                        config["checkpoint_name"]
                        .replace("best_checkpoint_", "")
                        .replace(".pt", "")
                        in x.name
                    )
                )
            ]
            old_config = config
            if len(configs_in_path) != 0:
                with open(configs_in_path[0], "r") as f:
                    config = yaml.safe_load(f)
                config["model_object"] = eval(config["model_object"])
                config["checkpoint_name"] = old_config["checkpoint_name"]
            
            trainer = BaseTrainer(**config, load=False)
            trainer.run_name = (
                config["checkpoint_name"]
                .replace("best_checkpoint_", "")
                .replace(".pt", "")
                + "_ft"
            )
            trainer.load(checkpoint_name=config["checkpoint_name"])
            trainer.train()
            trainer.submit_run()
        else:
            trainer = BaseTrainer(**config, load=False)
            trainer.run_name = (
                config["checkpoint_name"]
                .replace("best_checkpoint_", "")
                .replace(".pt", "")
            )
            trainer.load_checkpoint(config["checkpoint_name"])
            trainer.get_balanced_acc_val()
            trainer.submit_run()
    else:
        trainer = BaseTrainer(**config, load=True)
        trainer.train()
        trainer.submit_run()