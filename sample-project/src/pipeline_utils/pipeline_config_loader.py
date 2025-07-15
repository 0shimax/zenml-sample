import pathlib

import yaml


def read_yaml_file(filename) -> dict:
    with open(filename, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise RuntimeError from exc


def load_parameters(root_dir: str = "./config") -> dict:
    files = pathlib(root_dir).glob("*.yaml")
    return {file.stem: read_yaml_file(file) for file in files}
