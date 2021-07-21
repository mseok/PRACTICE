import sys

import hydra
from omegaconf import DictConfig, OmegaConf

import models.EGNN
import utils


@hydra.main(config_path="./conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    return


if __name__ == "__main__":
    # CONFIG_PATH = sys.argv[1]
    main()
