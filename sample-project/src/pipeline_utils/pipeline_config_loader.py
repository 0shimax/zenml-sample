import pathlib

import yaml


def read_yaml_file(filename) -> dict:
    with open(filename, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise RuntimeError from exc


def load_parameters(root_dir: str = "./configs") -> dict:
    files = pathlib.Path(root_dir).glob("*.yml")
    return {file.stem: read_yaml_file(file) for file in files}
