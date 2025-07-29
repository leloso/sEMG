import yaml

class Config:
    def __init__(self, config_file='config.yaml'):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        self.__dict__.update(config)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError(f"No attribute named '{name}' in the configuration.")

    def get(self, section, key, default=None):
        section_data = self.__dict__.get(section, {})
        return section_data.get(key, default)

# Load configuration from YAML file
cfg = Config()