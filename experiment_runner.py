import traceback
import sys
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="configs", config_name="config_dev", version_base=None)
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)

    try:
        # Lazy import to facilitate faster Hydra config loading
        from src.experiment import Experiment

        experiment = Experiment(config)
        experiment.run()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
