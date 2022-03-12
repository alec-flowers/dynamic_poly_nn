from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import sys


def get_confusion_matrix(loader, net, device):
    y_pred = []
    y_true = []
    net.eval()
    for (idx, data) in enumerate(loader):
        sys.stdout.write('\r [%d/%d]' % (idx + 1, len(loader)))
        sys.stdout.flush()
        img = data[0].to(device)
        label = data[1].to(device)
        with torch.no_grad():
            pred = net(img)

        _, predicted = pred.max(1)
        y_pred.extend(predicted)
        y_true.extend(label)
    print("Calculating Confusion Matrix")
    return plot_confusion_matrix(y_pred, y_true, loader.dataset.class_to_idx.keys())


def plot_confusion_matrix(targets, predicted, labels):
    conf_mat = confusion_matrix(targets, predicted)
    fig = sns.heatmap(conf_mat, annot=True,
                      cmap=sns.color_palette("light:#5A9", as_cmap=True),
                      cbar_kws={'label': 'count'}, fmt='g')
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    # plt.title("Confusion matrix for the " + title + " data")
    tick_marks = np.arange(len(labels)) + 0.5
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels, rotation=0)
    plt.tight_layout()
    return fig.get_figure()
