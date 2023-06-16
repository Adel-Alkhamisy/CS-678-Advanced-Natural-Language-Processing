import numpy as np
import matplotlib.pyplot as plt

# Learning Rate
def plot_lr():
    X = ['3e-1', '3e-3', '3e-5', '3e-7']
    exact_match = [1.28, 2.0, 59.76, 35.43]
    f1_score = [2.62, 3.25, 72.72, 50.81]

    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.2, exact_match, 0.4, label='Exact Match')
    plt.bar(X_axis + 0.2, f1_score, 0.4, label='F1 Score')

    plt.xticks(X_axis, X)
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.title("Learning Rate Training")
    plt.legend()
    plt.show()

plot_lr()

# Epochs
def plot_epochs():
    X = ['1', '3', '5', '7']
    exact_match = [57.94, 59.76, 57.82, 57.79]
    f1_score = [71.45, 72.72, 71.02, 71.46]

    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.2, exact_match, 0.4, label='Exact Match')
    plt.bar(X_axis + 0.2, f1_score, 0.4, label='F1 Score')

    plt.xticks(X_axis, X)
    plt.xlabel("No Of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Epochs Training")
    plt.legend()
    plt.show()

plot_epochs()

# Batch
def plot_batch():
    X = ['2', '4' , '6', '8', '16']
    exact_match = [57.19, 58.64, 58.34, 59.33, 59.76]
    f1_score = [69.92, 71.71, 71.85, 72.46, 72.72]

    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.2, exact_match, 0.4, label='Exact Match')
    plt.bar(X_axis + 0.2, f1_score, 0.4, label='F1 Score')

    plt.xticks(X_axis, X)
    plt.xlabel("Batch Size")
    plt.ylabel("Accuracy")
    plt.title("Batch Training")
    plt.legend()
    plt.show()

plot_batch()

# Pretrained Models
from textwrap import wrap
def plot_diff_models():
    X = ['bert_uncased_L-12_H-768_A-12', 'Roberta' , 'Bert-base-uncased', 'Camembert', 'AdamW']
    X = ['\n'.join(wrap(l, 15)) for l in X]
    exact_match = [49.11, 48.24, 49.18, 41.03, 59.52]
    f1_score = [63.03, 64.02, 62.49, 52.65, 72.29]

    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.2, exact_match, 0.4, label='Exact Match')
    plt.bar(X_axis + 0.2, f1_score, 0.4, label='F1 Score')
    plt.xticks(X_axis, X)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xlabel("Pretrained Model")
    plt.ylabel("Accuracy")
    plt.title("Epochs Training")
    plt.legend()
    plt.show()

plot_diff_models()