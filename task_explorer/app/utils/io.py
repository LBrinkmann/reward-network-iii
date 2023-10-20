import yaml


def load_yaml(filename):
    with open(filename) as f:
        data = yaml.safe_load(f)
    return data
