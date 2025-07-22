import hydra
from omegaconf import DictConfig
from src.models.train_model import ModelTrainer

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    ModelTrainer(cfg).run()

if __name__ == "__main__":
    main()
