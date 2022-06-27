from collections import defaultdict

import torch
import matplotlib.pyplot as plt
import re
import glob
import os
import numpy as np

from networks import nets
from load_data import load_dataset, load_testloader
from runner import test_to_analyze
from plots import plot_confusion_matrix, plot_image_grid, per_class_accuracy, plot_per_class_accuracy
from utils import get_activation, load_model, load_yaml, SCRATCH_PATH


def compare_two_nets_by_sample(path1, path2, config1, config2, **kwargs):
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    checkpoint1 = torch.load(path1, map_location=device)
    checkpoint2 = torch.load(path2, map_location=device)

    config1 = load_yaml(config1)
    config2 = load_yaml(config2)

    y_pred_1, y_true_1, labels, dataset = test_net(checkpoint1, config1, **kwargs)
    y_pred_2, y_true_2, _, _ = test_net(checkpoint2, config2, **kwargs)

    assert all(val1 == val2 for (val1, val2) in zip(y_true_1, y_true_2))

    title = f"{config1['dataset']['name']} x: {config1['model']['name']} y: {config2['model']['name']}"
    fig = plot_confusion_matrix(y_pred_1, y_pred_2, labels, title)
    plt.show()

    correct_1 = [0 if val1 == val2 else 1 for (val1, val2) in zip(y_pred_1, y_true_1)]
    correct_2 = [0 if val1 == val2 else 1 for (val1, val2) in zip(y_pred_2, y_true_2)]
    fig = plot_confusion_matrix(correct_1, correct_2, ["0", "1"], title)
    plt.show()

    difference_when_wrong(y_pred_1, y_pred_2, y_true_2)

    return y_pred_1, y_pred_2, y_true_2, dataset

def compare_n_nets_by_sample(paths, configs, **kwargs):
    assert len(paths) == len(configs)
    checkpoint_list = []
    config_list = []
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    for i in range(len(paths)):
        checkpoint_list.append(torch.load(paths[i], map_location=device))
        config_list.append(load_yaml(configs[i]))

    for k, v in kwargs.items():
        for config in config_list:
            if k in config['dataset']:
                config['dataset'][k] = v
            else:
                print(f"{k} not in config/dataset, no change occured")

    y_pred_list = []
    y_true_list = []
    for i in range(len(checkpoint_list)):
        print(f"File: {paths[i].split(os.sep)[-2]}")
        y_pred, y_true, labels, dataset = test_net(checkpoint_list[i], config_list[i], **kwargs)
        y_pred_list.append(y_pred)
        y_true_list.append(y_true)

    assert all(len(set(val)) == 1 for val in zip(*y_true_list))

    return dataset, y_true, y_pred_list


def difference_when_wrong(y_pred_1, y_pred_2, y_true):
    all_wrong = [1 if p1 != gt and p2 != gt else 0 for (p1, p2, gt) in zip(y_pred_1, y_pred_2, y_true)]
    ind_list = np.arange(0, len(all_wrong))
    chosen = ind_list[list(map(bool, all_wrong))]
    num_differ = sum([0 if p1 == p2 else 1 for (p1, p2) in zip(y_pred_1[chosen], y_pred_2[chosen])])
    print(f"{num_differ}/{sum(all_wrong)} {num_differ/sum(all_wrong):0.2f}% of misclassified were different.")


def diff_when_wrong_many(y_true, *args):
    # loop through number models +1 for none of them get it
    how_many_correct = []
    for i in range(len(args)+1):
        # check number of correct answers for each datapoint.
        # If none got it correct, then will be in first list, one correct in second list and so on.
        # another way is to loop through each list and each time append to correct list
        how_many_correct.append(
            list(map(bool,
                     [1 if sum([vals[0] == vals[i] for i in range(1, len(vals))]) - i == 0 else 0 for vals in zip(y_true, *args)])
                 )
        )

    ind_list = np.arange(0, len(y_true))
    correct_ind = []
    model_indices = {}
    for i, correct in enumerate(how_many_correct):
        correct_ind.append(ind_list[correct])
        print(f"{i} Models Correct: {sum(correct)} of {len(y_true)} is {sum(correct)/len(y_true)*100:0.1f}%")
        model_indices.update(model_combination_correct(ind_list[correct], y_true, *args))

    return correct_ind, model_indices


def model_combination_correct(ind, y_true, *args):
    counts = {}
    model_indices = defaultdict(list)
    for val in ind:
        model = []
        for i, pred in enumerate(args):
            if y_true[val] == pred[val]:
                model.append(i)

        counts[tuple(model)] = counts.get(tuple(model), 0) + 1
        model_indices[tuple(model)].append(val)

    for key, value in counts.items():
        if key:
            print(f"    - Models {key} Correct: {value} of {len(ind)} is {value/len(ind)*100:.1f}%")
    return model_indices


def plot_examples(dataset, y_pred_1, y_pred_2, y_true, show='all_wrong', num_images=9):
    all_wrong = [1 if p1 != gt and p2 != gt else 0 for (p1, p2, gt) in zip(y_pred_1, y_pred_2, y_true)]
    correct_1_other_wrong = [1 if p1 == gt and p2 != gt else 0 for (p1, p2, gt) in zip(y_pred_1, y_pred_2, y_true)]
    correct_2_other_wrong = [1 if p1 != gt and p2 == gt else 0 for (p1, p2, gt) in zip(y_pred_1, y_pred_2, y_true)]
    ind_list = np.arange(0, len(all_wrong))

    if show == 'all_wrong':
        chosen = ind_list[list(map(bool, all_wrong))]
    elif show == 'correct_1':
        chosen = ind_list[list(map(bool, correct_1_other_wrong))]
    elif show == 'correct_2':
        chosen = ind_list[list(map(bool, correct_2_other_wrong))]

    mapping = {v: k for k, v in dataset.class_to_idx.items()}
    img_list = []
    titles = []
    for i in np.random.choice(chosen, num_images, replace=False):
        img, label = dataset[i]
        assert label == y_true[i]
        img_list.append(img.permute((1, 2, 0)).squeeze())
        titles.append(f"p1:{mapping[y_pred_1[i]]} p2:{mapping[y_pred_2[i]]} correct:{mapping[y_true[i]]}")

    plot_image_grid(img_list, subplot_title=titles)


def test_net(checkpoint, config, register=[], activation=None, **kwargs):
    print('======================================================')
    print(f"Dataset: {config['dataset']['name']} Epochs: {config['training_info']['epochs']}")
    # Load Data in particular way

    train_dataset, test_dataset = load_dataset(
        **config['dataset']
    )
    # Initialize dataloaders
    test_loader = load_testloader(
        test_dataset,
        batch_size=config['dataloader']['batch_size'],
        num_workers=0,
        shuffle=False,
    )

    sample_shape = test_loader.dataset[0][0].shape
    # check image is square since using only 1 side of it for the shape
    assert (sample_shape[1] == sample_shape[2])
    image_size = sample_shape[1]
    num_classes = len(test_loader.dataset.classes)
    channels_in = sample_shape[0]

    net = load_model(
        config['model'],
        image_size=image_size,  # this parameter doesn't matter for Resnet
        num_classes=num_classes,
        channels_in=channels_in
    )
    net.load_state_dict(checkpoint['model_state_dict'])

    # add hooks
    for lay in register:
        if hasattr(net, lay):
            getattr(net, lay).register_forward_hook(get_activation(activation, lay))

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(f"Running on: {device}")

    if device == 'cuda':
        print(f"Num Devices: {torch.cuda.device_count()}")
        dev = torch.cuda.current_device()
        print(f"Device Name: {torch.cuda.get_device_name(dev)}")

    net.to(device)

    y_pred, y_true = test_to_analyze(net, test_loader, device)
    labels = test_loader.dataset.class_to_idx.keys()

    per_class_acc, avg_acc = per_class_accuracy(y_true, y_pred)
    plot_per_class_accuracy(y_true, y_pred, labels)
    print(f"Average Accuracy: {avg_acc:0.2f} \nPer Class Accuracy: {per_class_acc}")


    return y_pred, y_true, labels, test_dataset


if __name__ == '__main__':
    paths = list(filter(
        lambda x: not re.search('xxxx', x),
        glob.glob(f"{SCRATCH_PATH}/logs/CIFAR10/*low*/best_model", recursive=True)
    ))
    print(paths)

    configs = list(filter(
        lambda x: not re.search('xxxx', x),
        glob.glob(f"{SCRATCH_PATH}/logs/CIFAR10/*low*/r_ccp*", recursive=True)
    ))
    print(configs)

    dataset, y_true, y_pred = compare_n_nets_by_sample(paths, configs, apply_manipulation=None,)
    correct_ind = diff_when_wrong_many(y_true, *y_pred)
    b
