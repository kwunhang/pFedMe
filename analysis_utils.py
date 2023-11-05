import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def cifar_label_convert():
    pass

def plot_cm(true_labels, predict_labels, model_name):
    result_cm = confusion_matrix(true_labels, predict_labels)
    num_classes = 10
    class_names = [i for i in range(10)]
    fig, ax = plt.subplots()
    plt.imshow(result_cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"Confusion Matrix of {model_name}")
    ax.set_xticks(np.arange(result_cm.shape[1]))
    ax.set_yticks(np.arange(result_cm.shape[0]))
    ax.set_xticklabels(np.arange(len(np.unique(true_labels))))
    ax.set_yticklabels(np.arange(len(np.unique(true_labels))))
    tick_marks = numpy.arange(num_classes)
    thresh = result_cm.max() / 2.
    for i in range(result_cm.shape[0]):
        for j in range(result_cm.shape[1]):
            plt.text(j, i, format(result_cm[i, j]),
                    ha="center", va="center", size="small",
                    color="white" if  result_cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.colorbar()
    plt.show()
    plt.savefig(fname=("cifar_plot/cm_"+model_name))
    return plt

def computePRF(true_labels, predicted_labels,model_name):
    precision = precision_score(true_labels, predicted_labels, average=None)
    recall = recall_score(true_labels, predicted_labels, average=None)
    f1 = f1_score(true_labels, predicted_labels, average=None)

    # print("")
    # for i, label in enumerate(labels_noniid5):
    #     print(f"Metrics for class {label}:")
    #     print(f"Precision: {precision[i]}")
    #     print(f"Recall: {recall[i]}")
    #     print(f"F1 Score: {f1[i]}")
    #     print()
    
    label = np.unique(true_labels)
    num_x = np.arange(len(label))

    width = 0.2
    fig, ax = plt.subplots()
    rects1 = ax.bar(num_x - width, precision, width, label='Precision')
    rects2 = ax.bar(num_x, recall, width, label='Recall')
    rects3 = ax.bar(num_x + width, f1, width, label='F1 Score')

    ax.set_ylabel('Scores')
    ax.set_title(f"Precision, Recall, F1 Score of model {model_name}")
    ax.set_xticks(num_x)
    ax.set_xticklabels(label)
    ax.legend()

    plt.axis([-1, len(label), 0, 1])
    plt.savefig(fname=("cifar_plot/prf_"+model_name))
    plt.show()
