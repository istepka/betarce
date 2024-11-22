import hydra
from omegaconf import DictConfig, OmegaConf

from src.experimentsv3 import experiment



@hydra.main(config_path="configs", config_name="config_dev", version_base=None)
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    
    experiment(config)
    
    
if __name__ == "__main__":
    main()