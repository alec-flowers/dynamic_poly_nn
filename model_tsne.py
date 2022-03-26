import torch
import os
from torch import optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import datetime
from utils import MODEL_PATH, path_exist, count_parameters, load_checkpoint
from sklearn.manifold import TSNE
import numpy as np

from load_data import load_dataset, only_use_certain_class, OneClassDataset, load_testloader
from nets import CCP, NCP
from runner import train, test, train_profile, test_to_analyze
from plots import plot_per_class_accuracy, plot_tsne, plot_confusion_matrix, by_layer_tsne
import collections
import matplotlib.pyplot as plt
from test import test_net


if __name__ == '__main__':
    DATASETS = ['MNIST', "FashionMNIST", "CIFAR10"]
    chosen_dataset = DATASETS[2]
    n_degree = 2
    file = "sub2345_e100_20220326-094446.ckpt"

    # which layers to register hooks
    REGISTER = ["U1", "Id_U2"]

    CHECK_PATH = f"{MODEL_PATH}/{chosen_dataset}/{n_degree}/{file}"
    checkpoint = torch.load(CHECK_PATH)
    activation = collections.defaultdict(list)

    y_pred, y_true, labels = test_net(
        checkpoint,
        chosen_dataset=chosen_dataset,
        register=REGISTER,
        activation=activation
    )

    title = f"{chosen_dataset} degree: {n_degree}"
    plot_per_class_accuracy(y_true, y_pred, labels, title)
    fig = plot_confusion_matrix(y_true, y_pred, labels, title)
    plt.show()

    _ = by_layer_tsne(y_true, REGISTER, activation, labels, title)
