import hydra
from omegaconf import DictConfig
from src.models.drift_detector import DriftDetector

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    DriftDetector(cfg).run()

if __name__ == "__main__":
    main()
