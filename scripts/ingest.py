import hydra
from omegaconf import DictConfig
from src.data.ingestion import DataIngestor

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    DataIngestor(cfg).run()

if __name__ == "__main__":
    main()
