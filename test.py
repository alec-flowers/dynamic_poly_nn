import torch
import matplotlib.pyplot as plt
import collections
import re
import glob
import os
import numpy as np

import nets
from load_data import load_dataset, load_testloader
from runner import test_to_analyze
from plots import plot_confusion_matrix, plot_image_grid
from utils import MODEL_PATH, get_activation


def compare_two_nets_by_sample(path1, path2, chosen_dataset, **kwargs):
    checkpoint1 = torch.load(path1)
    checkpoint2 = torch.load(path2)

    assert checkpoint1["apply_manipulation"] == checkpoint2["apply_manipulation"]
    assert str(checkpoint1["transform"]) == str(checkpoint2["transform"])
    assert checkpoint1["ind_to_keep"] == checkpoint2["ind_to_keep"]
    assert checkpoint1["binary_class"] == checkpoint2["binary_class"]

    y_pred_1, y_true_1, labels, dataset = test_net(checkpoint1, chosen_dataset, **kwargs)
    y_pred_2, y_true_2, _, _ = test_net(checkpoint2, chosen_dataset, **kwargs)

    assert all(val1 == val2 for (val1, val2) in zip(y_true_1, y_true_2))

    title = f"{chosen_dataset} x: {checkpoint1['n_degree']} y: {checkpoint2['n_degree']}"
    fig = plot_confusion_matrix(y_pred_1, y_pred_2, labels, title)
    plt.show()

    title = f"{chosen_dataset} x: {checkpoint1['n_degree']} y: {checkpoint2['n_degree']}"
    correct_1 = [0 if val1 == val2 else 1 for (val1, val2) in zip(y_pred_1, y_true_1)]
    correct_2 = [0 if val1 == val2 else 1 for (val1, val2) in zip(y_pred_2, y_true_2)]
    fig = plot_confusion_matrix(correct_1, correct_2, ["0", "1"], title)
    plt.show()

    difference_when_wrong(y_pred_1, y_pred_2, y_true_2)

    return y_pred_1, y_pred_2, y_true_2, dataset


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
    for i, correct in enumerate(how_many_correct):
        correct_ind.append(ind_list[correct])
        print(f"{i} Models Correct: {sum(correct)} of {len(y_true)} is {sum(correct)/len(y_true)*100:0.1f}%")
        model_combination_correct(ind_list[correct], y_true, *args)

    return correct_ind


def model_combination_correct(ind, y_true, *args):
    counts = {}
    for val in ind:
        model = []
        for i, pred in enumerate(args):
            if y_true[val] == pred[val]:
                model.append(i)

        counts[tuple(model)] = counts.get(tuple(model), 0) + 1

    for key, value in counts.items():
        if key:
            print(f"    - Models {key} Correct: {value} of {len(ind)} is {value/len(ind)*100:.1f}%")
    return counts


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


def test_net(checkpoint, chosen_dataset, register=[], activation=None, **kwargs):
    # Load Data in particular way
    train_dataset, test_dataset = load_dataset(
        chosen_dataset,
        checkpoint["transform"],
        apply_manipulation=checkpoint["apply_manipulation"],
        binary_class=checkpoint["binary_class"],
        ind_to_keep=checkpoint["ind_to_keep"],
        **kwargs
    )
    # Initialize dataloaders
    test_loader = load_testloader(
        test_dataset,
        batch_size=checkpoint["batch_size"],
        num_workers=0,
        shuffle=False,
        seed=0,
    )

    print(f"Dataset: {type(test_loader.dataset)}")
    sample_shape = test_loader.dataset[0][0].shape
    # check image is square since using only 1 side of it for the shape
    assert (sample_shape[1] == sample_shape[2])
    image_size = sample_shape[1]
    n_classes = len(test_loader.dataset.classes)
    channels_in = sample_shape[0]

    net = getattr(nets, checkpoint.get("net_name", "CCP"))(
        checkpoint["hidden_size"],
        image_size=image_size,
        n_classes=n_classes,
        channels_in=channels_in,
        n_degree=checkpoint["n_degree"]
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

    y_pred, y_true = test_to_analyze(net, test_loader, device)
    labels = test_loader.dataset.class_to_idx.keys()

    return y_pred, y_true, labels, test_dataset


if __name__ == '__main__':
    paths = list(filter(
        lambda x: not re.search('xxxx', x),
        glob.glob(f"{MODEL_PATH}/CIFAR10/**/sub*.ckpt", recursive=True)
    ))
    print(paths)

    compare_two_nets_by_sample(paths[0], paths[1], paths[0].split(os.sep)[-3], color='uniform')
