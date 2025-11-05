import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import glob
from ruamel.yaml import YAML
from src.losses.EDIM_loss import EDIMLoss
from src.models.EDIM import EDIM
from src.utils.Card_dataloader import CardDataset
from src.trainer.EDIM_trainer import EDIMTrainer, freeze_grad_and_eval
from src.neural_networks.encoder import BaseEncoderSH
import h5py
from sklearn.model_selection import train_test_split


def run(
    xp_name: str,
    conf_path: str,
    data_base_folder: str,
    trained_enc_x_path: str,
    trained_enc_y_path: str,
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
        print("seed",seed)

    TRAINING_PARAM = conf["training_param"]
    MODEL_PARAM = conf["model_param"]
    LOSS_PARAM = conf["loss_param"]
    SHARED_PARAM = conf["shared_param"]
    batch_size = TRAINING_PARAM["batch_size"]

    kernel_size = 2
    img_size = (MODEL_PARAM["img_size"] - 3 + 2) // 1 + 1
    img_size1 = (img_size - kernel_size) // 2 + 1
    img_size2 = (img_size1 - kernel_size) // 2 + 1
    img_size3 = (img_size2 - kernel_size) // 2 + 1

    trained_enc_x = BaseEncoderSH(
        img_size1=img_size1,
        img_size2=img_size2,
        img_size3=img_size3,
        in_channels=3,
        num_filters=16,
        kernel_size=kernel_size,
        repr_dim=SHARED_PARAM["shared_dim"],
    )
    trained_enc_y = BaseEncoderSH(
        img_size1=img_size1,
        img_size2=img_size2,
        img_size3=img_size3,
        in_channels=3,
        num_filters=16,
        kernel_size=kernel_size,
        repr_dim=SHARED_PARAM["shared_dim"],
    )
    
    trained_enc_x.load_state_dict(torch.load(trained_enc_x_path))
    trained_enc_y.load_state_dict(torch.load(trained_enc_y_path))
    freeze_grad_and_eval(trained_enc_x)
    freeze_grad_and_eval(trained_enc_y)
    print("share encoder x:", trained_enc_x_path)
    print("share encoder y:", trained_enc_y_path)

    edim = EDIM(
        img_size=MODEL_PARAM["img_size"],
        channels=MODEL_PARAM["channels"],
        shared_dim=SHARED_PARAM["shared_dim"],
        exclusive_dim=MODEL_PARAM["exclusive_dim"],
        trained_encoder_x=trained_enc_x,
        trained_encoder_y=trained_enc_y,
        batch_size=TRAINING_PARAM["batch_size"],
    )
    loss = EDIMLoss(
        local_mutual_loss_coeff=LOSS_PARAM["local_mutual_loss_coeff"],
        global_mutual_loss_coeff=LOSS_PARAM["global_mutual_loss_coeff"],
        disentangling_loss_coeff=LOSS_PARAM["disentangling_loss_coeff"],
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
    batch_size = TRAINING_PARAM["batch_size"]
    epochs = TRAINING_PARAM["epochs"]
    trainer = EDIMTrainer(
        dataset_train=train_dataset,
        model=edim,
        loss=loss,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
    )
    trainer.train(epochs=epochs, xp_name=xp_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Learning Disentangled Representations via Mutual Information Estimation"
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

    parser.add_argument("--seed", nargs="?", type=int, default=142, help="Random seed")
    parser.add_argument(
        "--trained_enc_x_path",
        nargs="?",
        type=str,
        default=None,
        help="Pretrained shared encoder x",
    )
    parser.add_argument(
        "--trained_enc_y_path",
        nargs="?",
        type=str,
        default=None,
        help="Pretrained shared encoder y",
    )

    args = parser.parse_args()
    xp_name = args.xp_name
    conf_path = args.conf_path
    data_base_folder = args.data_base_folder
    seed = args.seed
    trained_enc_x_path = args.trained_enc_x_path
    trained_enc_y_path = args.trained_enc_y_path
    run(
        xp_name=xp_name,
        conf_path=conf_path,
        data_base_folder=data_base_folder,
        trained_enc_x_path=args.trained_enc_x_path,
        trained_enc_y_path=args.trained_enc_y_path,
        seed=seed,
    )
