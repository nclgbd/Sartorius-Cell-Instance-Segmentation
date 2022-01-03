import argparse
from Training import train

parser = argparse.ArgumentParser(description="Training script")

parser.add_argument(
    "--defaults_path",
    "-d",
    type=str,
    default="config/defaults.yaml",
    help="The path to the parameters.yaml file. Defaults to `config/defaults.yaml`",
)
args = parser.parse_args().__dict__


def main():
    DEFAULTS_PATH = args["defaults_path"]
    train(defaults_path=DEFAULTS_PATH)

if __name__ == "__main__":
    main()
