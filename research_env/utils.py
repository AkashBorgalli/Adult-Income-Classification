import yaml
from yaml.loader import SafeLoader

def read_yml(file_name):
    with open(file_name) as f:
        data = yaml.load(f, Loader=SafeLoader)
        return data
