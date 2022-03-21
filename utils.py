import pathlib
import os
import torch


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


def count_parameters(model):
    """
    Count parameters in a model
    """
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    return total_params


def load_checkpoint(net, checkpoint_path, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint Loaded - {checkpoint_path}")