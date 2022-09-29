"""
Runs a model on a single node across N-gpus.
"""
from ansible_vault import Vault
from pytorch_lightning import Trainer
import argparse
import collections
import os
import thor.builder
import torch
import yaml


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

    # ------------------------
    # 1 Init Lightning Model
    # ------------------------
    torch.backends.cudnn.benchmark = True
    model = thor.builder.build_model(cfg)
    if "pretrained_paths" in cfg["exp"]:
        model.load_pretrained(checkpoint_paths=cfg["exp"]["pretrained_paths"])

    # ------------------------
    # 2 Init Trainer
    # ------------------------
    trainer_args = cfg["exp"].get("trainer", {})
    trainer_args.update(
        {"logger": False, "checkpoint_callback": False, "early_stop_callback": False}
    )
    trainer = Trainer(**trainer_args)
    # ------------------------
    # 3 Start Training
    # ------------------------
    trainer.test(model)


if __name__ == "__main__":
    main()
