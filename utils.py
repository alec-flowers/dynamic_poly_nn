import pathlib
import os
import torch
import importlib
import numpy as np
import random
import logging
import yaml
import torch.nn as nn
import torch.nn.init as init
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator

logger = logging.getLogger(__name__)

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
SCRATCH_PATH = (REPO_ROOT / "scratch").absolute().resolve()

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
    logger.info(f"Checkpoint Loaded - {checkpoint_path}")


def save_checkpoint(net, opt, epoch, e_time, savepath, savename):
    logger.info(f"Saving Model - {savename} - Elapsed Time: {e_time:.0f} sec")
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
        logger.info('net.init None; returning')
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
                logger.info('Init style not recognized...')
        param_count += sum([p.data.nelement() for p in module.parameters()])
    return param_count


def create_logger(savepath, pathtime):
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(savepath, f'{pathtime}.log'), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logger.addHandler(console)

def load_yaml(path):
    try:
        with open(os.path.join(path), 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')
    return config

# From https://github.com/HaohanWang/HFC/blob/master/utility/frequencyHelper.py
def fft(img):
    return torch.fft.fft2(img)


def fftshift(img):
    return torch.fft.fftshift(fft(img))


def ifft(img):
    return torch.fft.ifft2(img)


def ifftshift(img):
    return ifft(torch.fft.ifftshift(img))


def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def mask_radial(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask


def filter_freq_3channel(img, mask):
    tmp = torch.zeros(img.shape)
    for j in range(img.shape[0]):
        fd = fftshift(img[j, :, :])
        fd = fd * mask
        img_low = ifftshift(fd)
        tmp[j, :, :] = torch.real(img_low)
    return tmp

# modified from https://laszukdawid.com/blog/2021/01/26/parsing-tensorboard-data-locally/
def convert_tb_data(root_dir, sort_by=None):
    """Convert local TensorBoard data into Pandas DataFrame.

    Function takes the root directory path and recursively parses
    all events data.
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.

    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.

    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.

    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.

    """

    def convert_tfevent(filepath, dirname):

        df =  pd.DataFrame([
            parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)
        ])
        df['run'] = dirname
        return df

    def parse_tfevent(tfevent):
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].simple_value),
        )

    columns_order = ['run', 'wall_time', 'name', 'step', 'value']

    out = []
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path, os.path.split(root)[-1]))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)

    return all_df.reset_index(drop=True)