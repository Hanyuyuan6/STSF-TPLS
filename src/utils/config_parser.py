import yaml  # yaml library, used to parse the yaml config files
from pathlib import Path  # Path class, makes file path handling easier

def load_config(config_path):
    p = Path(config_path)  # turn the config file path into a Path object
    with open(p, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)  # read the yaml config file and parse it into a dict
    if 'base' in cfg:
        base = load_config(p.parent / cfg['base'])  # recursively load the base config file, path relative to the current config file's directory
        deep_update(base, cfg)  # let the current config override the base config, giving config inheritance and merging
        cfg = base  # point cfg at the merged config
    return cfg  # return the final config dict

def deep_update(base, upd):
    for k, v in upd.items():  # walk every key/value pair in the update dict
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            deep_update(base[k], v)  # recurse when the matching key in base is a dict as well
        else:
            base[k] = v  # otherwise just overwrite the value in base with the one from the update dict