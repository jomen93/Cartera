#  ===========================================================================
#  @file:   metrics.py
#  @brief:  Module to measure the metrics of the model
#  @author: Johan Mendez
#  @date:   06/06/2021
#  @email:  johan.mendez@databiz.co
#  @status: Debug
#  @detail: version 1.0
#  ===========================================================================
import matplotlib.pyplot as plt
import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.metrics import (mean_squared_log_error, 
                             mean_squared_error,
                             r2_score,
                             mean_absolute_error,
                             accuracy_score,
                             recall_score,
                             f1_score, precision_score,
                             plot_confusion_matrix,
                             roc_curve, auc,confusion_matrix,
                             cohen_kappa_score
                            )

def metrics(y_test, y_pred):

    print("Accuracy:  {}".format(accuracy_score(y_test, y_pred)))
    print('Recall:    {}'.format(recall_score(y_test, y_pred)))
    print('F1-Score:  {}'.format(f1_score(y_test, y_pred)))
    print('Precision: {}'.format(precision_score(y_test, y_pred, zero_division="warn")))
    print('Kappa:     {}'.format(cohen_kappa_score(y_test, y_pred)))


def plot_confusion_matrix(y_test, y_pred):
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        cmap=plt.cm.Blues,
        fmt="g",
        linecolor="black",
        linewidths=0.2
        )
    plt.title("confusion Matrix Classifier")
    plt.savefig("confusion_matrix")

def plot_XGboost(results):
    epoch = len(results["validation_0"]["error"])
    x_axis = range(0, epoch)

    fig = plt.figure(figsize=(18,5))

    plt.subplot(121)
    plt.plot(x_axis, results["validation_0"]["auc"], "b--",label="Train")
    plt.plot(x_axis, results["validation_1"]["auc"], "lightcoral", label="Test")
    plt.legend()
    plt.ylabel("AUC")
    plt.xlabel("épocas")
    plt.title("XGBoost AUC")
    plt.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.9)

    plt.subplot(122)
    plt.plot(x_axis, results["validation_0"]["error"], "b--",label="Train")
    plt.plot(x_axis, results["validation_1"]["error"], "lightcoral",label="Test")
    plt.legend()
    plt.ylabel("AUC")
    plt.title("XGBoost Clasification Error")
    plt.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.9)

    plt.savefig("performance_curves")

def performance_curve(data):
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(121)
    ax.plot(data.history["accuracy"], "r-", label="Accuracy")
    ax.plot(data.history["val_accuracy"], "b-", label="val Accuracy")
    ax.grid(True)
    plt.legend()
    ax = fig.add_subplot(122)
    ax.plot(data.history["loss"], "r-", label="Loss")
    ax.plot(data.history["val_loss"], "b-", label="val Loss")
    ax.grid(True)
    plt.legend()
    plt.savefig("Neural Network performance")






