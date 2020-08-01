from matplotlib.pyplot import subplots, savefig
from matplotlib.pyplot import plot, legend
from seaborn import heatmap, set, axes_style
from sklearn.metrics import confusion_matrix
from os.path import join


def plot_confusion(y_true, y_pred, all_cat,
                   norm, path):
    """Plots the confusion matrix
    """
    set(font_scale=1.8)
    data = confusion_matrix(y_true, y_pred,
                            labels=all_cat,
                            normalize=norm)

    with axes_style(style="white"):
        subplots(figsize=(20, 15))
        ax = heatmap(data, annot=True,
                     vmax=1, vmin=0,
                     square=True, cmap='YlGnBu',
                     linewidths=0.5,
                     annot_kws={"size": 18},
                     fmt='.1f')

        ax.set_xticklabels(all_cat, rotation=90)
        ax.set_yticklabels(all_cat, rotation=0)

    with_norm = 'confusion_matrix_with_norm.png'
    without_norm = 'confusion_matrix_without_norm.png'
    fname = with_norm if norm else without_norm
    fname = join(path, fname)
    savefig(fname, dpi=300)


def plot_loss(train_loss, folder):
    plot(train_loss, label='Training loss')
    legend(frameon=False)
    savefig(fname=join(folder, "rnn_loss.png"),
            dpi=300)
