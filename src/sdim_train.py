import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import glob
from ruamel.yaml import YAML
from src.losses.SDIM_loss import SDIMLoss
from src.models.SDIM import SDIM
from src.trainer.SDIM_trainer import SDIMTrainer
from src.utils.Card_dataloader import CardDataset
from sklearn.model_selection import train_test_split
import h5py
from collections import Counter


def run(
        xp_name: str,
        conf_path: str,
        data_base_folder: str,
        seed: int = None,
):
    yaml = YAML(typ='safe', pure=True)  
    with open(conf_path, "r") as f:
        conf = yaml.load(f)  
    if seed is not None:
        seed = seed
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
        print(seed)

    TRAINING_PARAM = conf["training_param"]
    MODEL_PARAM = conf["model_param"]
    LOSS_PARAM = conf["loss_param"]
    batch_size = TRAINING_PARAM["batch_size"]

    sdim = SDIM(
        img_size=MODEL_PARAM["img_size"],
        channels=MODEL_PARAM["channels"],
        shared_dim=MODEL_PARAM["shared_dim"],
        switched=MODEL_PARAM["switched"],
        batch_size=batch_size
    )
    loss = SDIMLoss(
        local_mutual_loss_coeff=LOSS_PARAM["local_mutual_loss_coeff"],
        global_mutual_loss_coeff=LOSS_PARAM["global_mutual_loss_coeff"],
        shared_loss_coeff=LOSS_PARAM["shared_loss_coeff"],
    )

    # Load data
    dataset_path_train = os.path.join(data_base_folder, 'Card_pair_dataset.h5')
    if not os.path.isfile(dataset_path_train):
        raise FileNotFoundError(
            f"{dataset_path} not found. Please place the dataset in this directory.")

    print("LoveDA loading")
    with h5py.File(dataset_path_train, 'r') as dataset:
        images_pair_train = dataset['images'][:]
        factors_pair_train = dataset['labels'][:]
    print("LoveDA finish")
    
    print("dataset size:",images_pair_train.shape)
    print(factors_pair_train.shape)

    train_dataset = CardDataset(images_pair_train, factors_pair_train)

    device = TRAINING_PARAM["device"]
    learning_rate = TRAINING_PARAM["learning_rate"]
    epochs = TRAINING_PARAM["epochs"]
    trainer = SDIMTrainer(
        dataset_train=train_dataset,
        model=sdim,
        loss=loss,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
    )
    trainer.train(epochs=epochs, xp_name=xp_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learning Disentangled Representation via Robust Optimal Transport"
    )
    parser.add_argument(
        "--xp_name",
        nargs="?",
        type=str,
        default="Shared_training",
        help="Mlflow experiment name",
    )
    parser.add_argument(
        "--conf_path", nargs="?", type=str, default=None, help="Configuration file"
    )
    parser.add_argument(
        "--data_base_folder", nargs="?", type=str, default=None, help="Data folder"
    )
    parser.add_argument("--seed", nargs="?", type=int, default=144, help="Random seed")

    args = parser.parse_args()
    xp_name = args.xp_name
    conf_path = args.conf_path
    data_base_folder = args.data_base_folder
    seed = args.seed

    run(
        xp_name=xp_name,
        conf_path=conf_path,
        data_base_folder=data_base_folder,
        seed=seed,
    )
