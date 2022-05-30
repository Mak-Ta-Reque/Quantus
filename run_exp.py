import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
from yaml.loader import SafeLoader

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> dict:
    config = yaml.load(OmegaConf.to_yaml(cfg),  Loader=SafeLoader)["config"]
    print(config)






if __name__ == "__main__":
    main()