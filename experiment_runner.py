import traceback
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

from src.experimentsv3 import experiment



@hydra.main(config_path="configs", config_name="config_dev", version_base=None)
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    
    try:
        experiment(config)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise
    
    
if __name__ == "__main__":
    main()