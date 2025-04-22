import os
import yaml
import pprint
from typing import Dict, Any


class Config:
    """
    Configuration handler for the project.
    """

    def __init__(self, config_path: str = None, config_dict: Dict[str, Any] = None):
        """
        Initialize configuration handler.

        Args:
            config_path (str, optional): Path to YAML config file
            config_dict (Dict[str, Any], optional): Configuration as a dictionary
        """
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config_dict is not None:
            self.config = config_dict
        else:
            self.config = {}

    def __getitem__(self, key):
        return self.config[key]

    def __contains__(self, key):
        return key in self.config

    def get(self, key, default=None):
        """
        Get a value from config with a default.

        Args:
            key (str): Key to look up in the config
            default (Any, optional): Default value if key doesn't exist

        Returns:
            Any: Value from config or default
        """
        return self.config.get(key, default)

    def update(self, config_dict):
        """
        Update config with new values.

        Args:
            config_dict (Dict[str, Any]): Dictionary to update config with
        """
        self.config.update(config_dict)

    def save(self, save_path):
        """
        Save config to YAML file.

        Args:
            save_path (str): Path to save the config
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def __repr__(self):
        return pprint.pformat(self.config)