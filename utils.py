import pathlib
import os


def path_exist(path):
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")


REPO_ROOT = pathlib.Path(__file__).absolute().parents[0].resolve()
assert (REPO_ROOT.exists())

MODEL_PATH = (REPO_ROOT / "models").absolute().resolve()
path_exist(MODEL_PATH)