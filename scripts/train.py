"""
Runs a model on a single node across N-gpus.
"""
from ansible_vault import Vault
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from thor.modules.callback.model_checkpoint import MultipleModelCheckpoint
import argparse
import collections
import numpy as np
import os
import random
import shutil
import thor.builder
import torch
import yaml


def set_random_seed(seed):
    """
    To be able to reproduce exps on reload
    Is pytorch dataloader with multi-threads deterministic ?
    cudnn may not be deterministic anyway
    """
    torch.manual_seed(seed)  # on CPU and GPU
    np.random.seed(seed)  # useful ? not thread safe
    random.seed(seed)  # useful ? thread safe


def fetch_last_checkpoint(ckpt_dir):
    """
    Args:
        experiment_dir: [str], the complete directory, which has a folder called "checkpoints" in it
    Returns:
        last_checkpoint_filename: [str] or None
    """
    # check checkpoints folder
    all_last_ckpts = []
    for ckpt in os.listdir(ckpt_dir):
        if "last_epoch=" in ckpt:
            all_last_ckpts.append(ckpt)
    if len(all_last_ckpts) > 0:
        # sort and get the last one
        last_ckpt = sorted(
            all_last_ckpts,
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("=")[-1]),
        )[-1]
        return os.path.join(ckpt_dir, last_ckpt)
    return None


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def main():
    """
    Main training routine specific for this project
    :param hparams:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", type=str, help="Yaml name")
    parser.add_argument(
        "-a",
        "--abstract",
        type=str,
        help="Abstract Yaml name",
        default="exp_config/tmuh-brain_tumor/abstract.yaml",
    )
    parser.add_argument("-r", "--resume", type=str, help="Checkpoint filepath to resume")
    args, unparsed = parser.parse_known_args()

    # ------------------------
    # 0 Parse Yaml and Load Args
    # ------------------------

    yaml_dir = os.path.join(".", args.out)

    # Load yaml for production version
    if os.getenv("PROD", None):

        yaml_abstract_dir = os.path.join(".", args.abstract)

        if not os.path.exists(yaml_abstract_dir):
            raise FileNotFoundError("Abstract file missing. Please contact with Taiwan AI Labs.")

        vault = Vault("w@_bro2rlrlremofit#ip*koprOTafo7a-huv@fEcathochuDre50Zl9uspuMIph")
        cfg = vault.load(open(yaml_abstract_dir).read())

        with open(yaml_dir, "r") as f:
            cfg_tmp = yaml.load(f, yaml.Loader)

        cfg = update(cfg, cfg_tmp)
        del cfg_tmp

    else:

        with open(yaml_dir, "r") as f:
            cfg = yaml.load(f, yaml.Loader)

        import pprint

        pprint.pprint(cfg)

    if "random_seed" in cfg["exp"].keys():
        set_random_seed(cfg["exp"]["random_seed"])

    trainer_args = cfg["exp"].get("trainer", {})
    log_args = cfg["exp"].get("logger", {})
    checkpoint_args = cfg["exp"].get("checkpointer", {})

    # ------------------------
    # 1 Init Checkpoint Callback and Logger
    # ------------------------

    trainer_args["default_root_dir"] = os.path.join(
        os.getcwd(), trainer_args["default_root_dir"] + "_exp"
    )
    log_args["save_dir"] = trainer_args["default_root_dir"]
    logger = TensorBoardLogger(**log_args)
    logger_exp_path = logger.experiment.log_dir

    shutil.copyfile(yaml_dir, os.path.join(logger_exp_path, "exp.yaml"))

    ckpt_path = os.path.join(logger_exp_path, "checkpoints")
    checkpoint_args["filepath"] = ckpt_path
    checkpoint_args["heavy_val_freq"] = cfg["exp"].get("heavy_val_freq", 1)
    checkpoint_callback = MultipleModelCheckpoint(**checkpoint_args)

    print("Now run exp {}".format(logger.experiment.log_dir))
    print("Save in {}".format(os.path.join(os.getcwd(), checkpoint_args["filepath"])))

    # ------------------------
    # 2 Init Lightning Model
    # ------------------------
    torch.backends.cudnn.benchmark = True
    # assignself. experiment path to model
    cfg["logger_exp_path"] = logger_exp_path
    model = thor.builder.build_model(cfg)
    if "pretrained_paths" in cfg["exp"]:
        model.load_pretrained(checkpoint_paths=cfg["exp"]["pretrained_paths"])

    # ------------------------
    # 3 Init Trainer
    # ------------------------
    trainer_args["checkpoint_callback"] = checkpoint_callback
    trainer_args["logger"] = logger

    # ------------------------
    # 4 Check Resume
    # ------------------------
    if "resume_from_checkpoint" not in trainer_args:
        resume_ckpt = args.resume
        if resume_ckpt is None:
            # args.resume is None
            resume_ckpt = fetch_last_checkpoint(ckpt_path)
        trainer_args["resume_from_checkpoint"] = resume_ckpt

    model.current_epoch = (
        -1
        if trainer_args["resume_from_checkpoint"] is None
        else torch.load(trainer_args["resume_from_checkpoint"], "cpu")["epoch"] - 1
    )

    # ------------------------
    # 5 Start Training
    # ------------------------
    trainer = Trainer(**trainer_args)
    trainer.fit(model)


if __name__ == "__main__":
    main()
