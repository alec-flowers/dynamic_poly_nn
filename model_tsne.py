import torch
import os
from torch import optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import datetime
from utils import MODEL_PATH, path_exist, count_parameters, load_checkpoint
from sklearn.manifold import TSNE
import numpy as np
import glob
import re

from load_data import load_dataset, only_use_certain_class, OneClassDataset, load_testloader, FewClassDataset
from nets import CCP, NCP
from runner import train, test, train_profile, test_to_analyze
from plots import plot_per_class_accuracy, plot_tsne, plot_confusion_matrix, by_layer_tsne
import collections
import matplotlib.pyplot as plt
from test import test_net


if __name__ == '__main__':
    for file in filter(
            lambda x: not re.search('xxxx', x),
            glob.glob(f"{MODEL_PATH}/MNIST/**/spec_tl*.ckpt", recursive=True)
    ):
        color = None
        chosen_dataset = file.split(os.sep)[-3]
        n_degree = file.split(os.sep)[-2]

        # which layers to register hooks
        REGISTER = []

        checkpoint = torch.load(file)
        activation = collections.defaultdict(list)

        y_pred, y_true, labels, _ = test_net(
            checkpoint,
            chosen_dataset=chosen_dataset,
            register=REGISTER,
            activation=activation,
            color=color,
            apply_manipulation=checkpoint["apply_manipulation"]
        )

        title = f"{chosen_dataset} degree: {n_degree}"
        plot_per_class_accuracy(y_true, y_pred, labels, title)
        fig = plot_confusion_matrix(y_true, y_pred, labels, title)
        plt.show()

        # _ = by_layer_tsne(y_true, REGISTER, activation, labels, title)
