from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import sys
import math


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

    # For tensorboard output
    return fig.get_figure()


def prepare_images(dataset, indices):
    # mapping so can give an number and will give back description
    mapping = {v: k for k, v in dataset.class_to_idx.items()}
    img_list = []
    label_list = []
    for i in indices:
        img, label = dataset[i]
        # put into proper form for plt.imshow
        img_list.append(img.permute((1, 2 ,0)).squeeze())
        label_list.append(mapping[label])
    return img_list, label_list


def plot_image_grid(images, title: str = "", subplot_title: list =[]):
    # images must be ready to be plotted by plt, that means (M, N, Channels)
    n_images = len(images)
    plt.figure(figsize=(10, 10))
    for i in range(n_images):
        image = images[i]
        if subplot_title:
            plt.subplot(math.ceil(np.sqrt(n_images)), math.ceil(np.sqrt(n_images)), i + 1, ).set_title(f"{subplot_title[i]}")
        else:
            plt.subplot(math.ceil(np.sqrt(n_images)), math.ceil(np.sqrt(n_images)), i + 1,)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap='gray')
    plt.tight_layout()
    plt.suptitle(title, size=16)
    plt.show()
