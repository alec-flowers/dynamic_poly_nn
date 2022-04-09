import pathlib
import os
import torch
import importlib
import numpy as np
import random

import torch.nn as nn
import torch.nn.init as init

def path_exist(*paths):
    for path in paths:
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print(f"The new directory is created: {path}")


REPO_ROOT = pathlib.Path(__file__).absolute().parents[0].resolve()
assert (REPO_ROOT.exists())

# MODEL_PATH = (REPO_ROOT / "models").absolute().resolve()
# path_exist(MODEL_PATH)
# WRITER_PATH = (REPO_ROOT / "runs").absolute().resolve()
# path_exist(WRITER_PATH)
NETWORK_PATH = (REPO_ROOT / "networks").absolute().resolve()
assert (NETWORK_PATH.exists()), "Create a network folder in repository root."
CONFIGS_PATH = (REPO_ROOT / "configs").absolute().resolve()
assert (CONFIGS_PATH.exists()), "Create a configs folder in repository root."


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
    return checkpoint
    print(f"Checkpoint Loaded - {checkpoint_path}")


def save_checkpoint(net, opt, epoch, e_time, savepath, savename):
    print(f"Saving Model - {savename} - Elapsed Time: {e_time:.0f} sec")
    torch.save(
        {
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'epoch': epoch,
            "elapsed_time": e_time,
        }, os.path.join(savepath, savename))


def get_activation(activation, name):
    def hook(inst, inp, out):
        activation[name].append(out.detach().numpy())
    return hook


def load_model(m_config, **kwargs):
    net = getattr(
        importlib.import_module(m_config['fn']), m_config['name']
    )
    return net(**m_config['args'], **kwargs)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_weights(net, gain_ortho=1, gain_xavier=1, mean_norm=0, std_norm=1, **kwargs):
    """
    Initialize the pytorch weights of a network. It uses a single format of initialization
    saved in net.init.
    :param net: pytorch type network; it's weights will be initialized.
    :param gain_ortho: float; the gain in the orthogonal initialization.
    :param gain_xavier: float; the gain in the xavier initialization.
    :param mean_norm: float; the mean in the gaussian initialization.
    :param std_norm: float; the std in the gaussian initialization.
    :return: int; The number of parameters.
    """

    if not getattr(net, 'init'):
        print('[Utils] net.init None; returning')
        return
    param_count = 0
    for module in net.modules():
        if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or
                isinstance(module, nn.Embedding) or isinstance(module, nn.ConvTranspose2d)):
            if net.init in ['gaussian', 'normal', 'N', 0]:
                init.normal_(module.weight, mean_norm, std_norm)
            elif net.init in ['glorot', 'xavier', 1]:
                init.xavier_uniform_(module.weight, gain=gain_xavier)
            elif net.init in ['ortho', 2]:
                init.orthogonal_(module.weight, gain=gain_ortho)
            elif net.init in ['kaiming', 'he_normal', 3]:
                init.kaiming_normal_(module.weight)
            elif net.init in ['kaimingun', 'he_uniform', 4]:
                init.kaiming_uniform_(module.weight)
            else:
                print('Init style not recognized...')
        param_count += sum([p.data.nelement() for p in module.parameters()])
    return param_count