import yaml
from typing import Dict, Any

class ConfigWrapper:
    def __init__(self, config_file: str):
        """
        Initialize ConfigWrapper with the path to the YAML configuration file.

        Args:
            config_file (str): Path to the YAML configuration file.
        """
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def get_general_config(self) -> Dict[str, Any]:
        """
        Get the general configuration settings.

        Returns:
            dict: General configuration settings.
        """
        return self.config.get('general', {})

    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        Get the configuration settings for a specific model type.

        Args:
            model_type (str): Type of the model (e.g., 'mlp', 'rf').

        Returns:
            dict: Configuration settings for the specified model type.
        """
        return self.config.get(model_type, {})

    def get_config_by_key(self, key: str) -> Any:
        """
        Get configuration setting by key.

        Args:
            key (str): Key of the configuration setting.

        Returns:
            Any: Configuration setting value corresponding to the key.
        """
        if key in self.config:
            return self.config[key]
        else:
            # Go through the entire configuration to find the key
            for _, value in self.config.items():
                if isinstance(value, dict):
                    if key in value:
                        return value[key]
        return None

    def get_entire_config(self) -> Dict[str, Any]:
        """
        Get the entire configuration.

        Returns:
            dict: Entire configuration settings.
        """
        return self.config

if __name__ == '__main__':
    # Example usage
    config_file = "config.yml"
    config = ConfigWrapper(config_file)

    # Get general configuration
    general_config = config.get_general_config()
    print("General Configuration:")
    print(general_config)

    # Get MLP model configuration
    mlp_config = config.get_model_config('mlp')
    print("\nMLP Configuration:")
    print(mlp_config)

    # Get RF model configuration
    rf_config = config.get_model_config('rf')
    print("\nRF Configuration:")
    print(rf_config)

    # Get configuration by key
    data_path = config.get_config_by_key('general')['data_path']
    print("\nData Path:", data_path)

    # Get entire configuration
    entire_config = config.get_entire_config()
    print("\nEntire Configuration:")
    print(entire_config)
