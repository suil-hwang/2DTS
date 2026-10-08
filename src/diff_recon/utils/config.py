import yaml
import json
from argparse import Namespace


class Config(Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return None

    def __str__(self):
        descriptor = f" {self._config_path} " if self._config_path is not None else " Config File "
        sep_len = max((100 - len(descriptor)) // 2, 10)
        banner_len = len(descriptor) + sep_len * 2
        config_str = "\n" + "=" * sep_len + descriptor + "=" * sep_len + "\n"
        config_str += yaml.safe_dump(configToDict(self), indent=4, sort_keys=False)
        config_str += "=" * banner_len + "\n"
        return config_str

    def __copy__(self):
        return dictToConfig(configToDict(self, ignore_private=False))


def dictToConfig(d):
    if isinstance(d, dict):
        return Config(**{k: dictToConfig(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dictToConfig(value) for value in d]
    elif isinstance(d, tuple):
        return tuple(dictToConfig(value) for value in d)
    else:
        return d


def configToDict(c, ignore_private=True):
    if isinstance(c, Config):
        if ignore_private:
            d = {k: c.__dict__[k] for k in c.__dict__.keys() if not k.startswith("_")}
        else:
            d = {k: c.__dict__[k] for k in c.__dict__.keys()}
        for k in d.keys():
            d[k] = configToDict(d[k], ignore_private)
        return d
    elif isinstance(c, list):
        return [configToDict(value, ignore_private) for value in c]
    elif isinstance(c, tuple):
        return tuple(configToDict(value, ignore_private) for value in c)
    else:
        return c


def loadConfig(config_path: str) -> Config:
    config_path = str(config_path)
    if not config_path.endswith((".json", ".yaml")):
        raise ValueError(f"Unknown config file type: {config_path}")
    with open(config_path, encoding="utf-8") as config_file:
        if config_path.endswith(".json"):
            config_dict = json.load(config_file)
        elif config_path.endswith(".yaml"):
            config_dict = yaml.safe_load(config_file)

    if not isinstance(config_dict, dict):
        raise ValueError("Config file must contain a mapping")

    config = dictToConfig(config_dict)
    config.__setattr__("_config_path", config_path)
    return config


def saveConfig(config: Config, config_path: str):
    config_path = str(config_path)
    if not config_path.endswith((".json", ".yaml")):
        raise ValueError(f"Unknown config file type: {config_path}")
    config_dict = configToDict(config)

    with open(config_path, "w", encoding="utf-8") as config_file:
        if config_path.endswith(".json"):
            json.dump(config_dict, config_file, indent=4)
        elif config_path.endswith(".yaml"):
            yaml.safe_dump(config_dict, config_file, indent=4, sort_keys=False)
