from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import sys
import math
import time


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

    # Bring everything to cpu for numpy to work
    y_pred = torch.tensor(y_pred, device='cpu').numpy()
    y_true = torch.tensor(y_true, device='cpu').numpy()
    print("Calculating Confusion Matrix")
    return plot_confusion_matrix(y_pred, y_true, loader.dataset.class_to_idx.keys())


def plot_confusion_matrix(targets, predicted, labels, title):
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
    if title:
        plt.title(f"Confusion Matrix {title[0]} degree: {title[1]}")
    plt.tight_layout()

    # For tensorboard output
    return fig.get_figure()


def per_class_accuracy(targets, predicted):
    conf_mat = confusion_matrix(targets, predicted)
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    avg_acc = accuracy_score(targets, predicted)
    return conf_mat.diagonal(), avg_acc


def plot_per_class_accuracy(targets, predicted, labels, title=None):
    per_class, avg_acc = per_class_accuracy(targets, predicted)
    fix, ax = plt.subplots(figsize=(12, 6))
    p1 = ax.bar(labels, per_class, align='center', alpha=0.5)
    ax.set_ylabel('Test Accuracy')
    plt.ylim([0, 1])
    if title:
        ax.set_title(f"Per Class Accuracy {title[0]} degree: {title[1]}")
    ax.bar_label(p1, fmt='%.3f', fontsize=11)
    plt.axhline(y=avg_acc, color='r', linestyle='-')
    plt.text(1/len(per_class)-1, avg_acc, f"{avg_acc:.3f}", rotation=0, color='r')
    plt.show()
    print(f"Test Set Average Accuracy: {avg_acc}")


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


def apply_pca(data, n_components):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)
    print('Cumulative explained variation for 50 principal components: {}'.format(
        np.sum(pca.explained_variance_ratio_)))

    return pca_result


def apply_tsne(data):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    return tsne_results


def plot_tsne(x, y, true_classes, n_colors, title=None):
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=x, y=y,
        hue=true_classes,
        palette=sns.color_palette("hls", n_colors=n_colors),
        legend="full",
        alpha=0.3
    )
    if title:
        plt.title(f"TSNE Plot for {title[0]} degree: {title[1]} Layer: {title[2]}")
    plt.show()
