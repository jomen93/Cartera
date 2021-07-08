import matplotlib.pyplot as plt
import seaborn as sns 

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