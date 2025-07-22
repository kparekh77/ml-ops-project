import os
import hydra
from hydra.utils import to_absolute_path, get_original_cwd
from omegaconf import DictConfig

from src.features.engineering import FeatureEngineer

def _resolve_paths(cfg_node: DictConfig):
    """
    Walk through cfg_node (and nested DictConfigs), and for any string
    whose key ends in path/file/dir, convert it into an absolute path
    under the original project root.
    """
    for key, val in cfg_node.items():
        if isinstance(val, str) and key.lower().endswith(("path", "file", "dir")):
            cfg_node[key] = to_absolute_path(val)
        elif isinstance(val, DictConfig):
            _resolve_paths(val)

@hydra.main(config_path="../config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    os.chdir(get_original_cwd())

    _resolve_paths(cfg)
    FeatureEngineer(cfg).run()

if __name__ == "__main__":
    main()
