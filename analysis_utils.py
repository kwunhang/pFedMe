import matplotlib
matplotlib.use('Agg')

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import h5py

def plot_function(true_label, predict_label, graph_name):
        plot_cm(true_label,predict_label, graph_name)
        computePRF(true_label,predict_label, graph_name)
        assert len(true_label)== len(predict_label)
        accuracy = ((np.array(true_label) == np.array(predict_label)).sum())/len(true_label)
        print("{} acc:".format(graph_name) ,accuracy)
        return "{} acc:".format(graph_name) + str(accuracy)

def cifar_label_convert():
    pass

def plot_cm(true_labels, predict_labels, model_name):
    result_cm = confusion_matrix(true_labels, predict_labels)
    num_classes = 10
    class_names = [i for i in range(10)]
    
    plot_path = os.getenv('SAVE_PLOT_PATH')
    if plot_path == None or plot_path == "":
        plot_path = "/kaggle/working/pFedMe/cifar_plot"
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
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
    plt.savefig(fname=(plot_path +"/cm_"+model_name))
    return plt

def computePRF(true_labels, predicted_labels, model_name):
    precision = precision_score(true_labels, predicted_labels, average=None)
    recall = recall_score(true_labels, predicted_labels, average=None)
    f1 = f1_score(true_labels, predicted_labels, average=None)
    
    plot_path = os.getenv('SAVE_PLOT_PATH')
    
    print("debug for plot_path:", plot_path)
    if plot_path == None or plot_path == "":
        plot_path = "/kaggle/working/pFedMe/cifar_plot"
    if not os.path.exists(plot_path):
            os.makedirs(plot_path)
            
    print("debug for plot_path2:", plot_path)
    
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
    plt.savefig(fname=(plot_path+"/prf_"+model_name))
    plt.show()

def compare_different_PRF(algorithms, true_labels_list, predicted_labels_list, pm_steps):
    if pm_steps == "":
        pm_steps = "Global Model"
    
    # Define a dictionary to hold the precision, recall, and f1 score for each algorithm
    performance_metrics = {alg: {'precision': [], 'recall': [], 'f1': []} for alg in algorithms}

    # Compute PRF for each algorithm
    for model_name, true_labels, predicted_labels in zip(algorithms, true_labels_list, predicted_labels_list):
        precision = precision_score(true_labels, predicted_labels, average=None)
        recall = recall_score(true_labels, predicted_labels, average=None)
        f1 = f1_score(true_labels, predicted_labels, average=None)

        performance_metrics[model_name]['precision'] = precision
        performance_metrics[model_name]['recall'] = recall
        performance_metrics[model_name]['f1'] = f1
        computePRF(true_labels, predicted_labels, model_name)  # Call the computePRF function for each algorithm

    # Plotting the comparison
    plot_path = os.getenv('SAVE_PLOT_PATH')
    
    print("debug for plot_path:", plot_path)
    if plot_path == None or plot_path == "":
        plot_path = "/kaggle/working/pFedMe/cifar_plot"
    if not os.path.exists(plot_path):
            os.makedirs(plot_path)

    labels = np.unique(np.concatenate(true_labels_list))  # assuming all algorithms have the same label set
    num_x = np.arange(len(labels))

    width = 0.2  # Width of the bars
    num_algorithms = len(algorithms)
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model_name in enumerate(algorithms):
        offset = (i - num_algorithms / 2) * width + width / 2

        # Retrieve the precision, recall, and f1 scores for the current algorithm
        precision = performance_metrics[model_name]['precision']
        recall = performance_metrics[model_name]['recall']
        f1 = performance_metrics[model_name]['f1']

        # Plot the bars for precision, recall, and f1 scores
        rects1 = ax.bar(num_x + offset, precision, width, label='Precision - ' + model_name)
        rects2 = ax.bar(num_x + offset + width, recall, width, label='Recall - ' + model_name)
        rects3 = ax.bar(num_x + offset + 2 * width, f1, width, label='F1 Score - ' + model_name)

    ax.set_ylabel('Scores')
    ax.set_title('Precision, Recall, and F1 Score Comparison')
    ax.set_xticks(num_x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.axis([-1, len(labels), 0, 1.5])
    plt.savefig(fname=os.path.join(plot_path, "prf_comparison_{}".format(pm_steps)))
    plt.show()

def plot_train_results(h5_path, model_name):
    with h5py.File(h5_path, 'r') as hf:
        # Load the data from the h5 file using the keys
        rs_glob_acc = hf['rs_glob_acc'][:]
        rs_train_acc = hf['rs_train_acc'][:]
        rs_train_loss = hf['rs_train_loss'][:]
    
    # Plot train loss
    plt.figure(figsize=(10, 5))
    plt.plot(rs_train_loss, label='Train Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('{} Train Loss'.format(model_name))
    plt.legend()
    plt.show()
    plt.savefig(fname=("Cifar_per_plot/loss_"+model_name))

    # Plot train accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(rs_train_acc, label='Train Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('{} Train Accuracy'.format(model_name))
    plt.legend()
    plt.show()
    plt.savefig(fname=("Cifar_per_plot/acc_"+model_name))

    # Plot global accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(rs_glob_acc, label='Global Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('{} Global Accuracy'.format(model_name))
    plt.legend()
    plt.show()
    plt.savefig(fname=("Cifar_per_plot/global_acc_"+model_name))
    
    return plt

def compare_model_PRF_function(true_label_list, predict_label_list, graph_name, analysis_files):
    assert len(true_label_list) == len(predict_label_list) == len(analysis_files), "Lists must have the same length."

    # Initialize dictionaries to hold precision, recall, and f1 scores for each model
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    
    model_names = []
    
    for analysis_file in analysis_files:
        # analysis_file remove the file type
        file_name, file_extension = os.path.splitext(analysis_file)
        if file_extension.lower() in (".pt", ".h5"):
            model_names.append(file_name)
    
    # Calculate metrics for each model
    for i, model_name in enumerate(model_names):
        precision_dict[model_name] = precision_score(true_label_list[i], predict_label_list[i], average='weighted')
        recall_dict[model_name] = recall_score(true_label_list[i], predict_label_list[i], average='weighted')
        f1_dict[model_name] = f1_score(true_label_list[i], predict_label_list[i], average='weighted')

    # Plotting
    plot_path = os.getenv('SAVE_PLOT_PATH', "plot")
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # The number of models to compare
    num_models = len(model_names)

    width = 0.2  # Width of the bars
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate the offset for bar's x-coordinate for each model
    offsets = np.linspace(-width, width, num_models)
    bar_width = width / num_models  # Adjust the bar width based on the number of models

    # Iterate over each model and plot the precision, recall, and F1 score
    for i, model_name in enumerate(model_names):
        bar_positions = np.arange(len(precision_dict[model_name])) + offsets[i]
        # Plot each metric
        ax.bar(bar_positions, precision_dict[model_name], bar_width, label=f'Precision of {model_name}')
        ax.bar(bar_positions, recall_dict[model_name], bar_width, bottom=precision_dict[model_name], label=f'Recall of {model_name}')
        ax.bar(bar_positions, f1_dict[model_name], bar_width, bottom=[u+v for u, v in zip(precision_dict[model_name], recall_dict[model_name])], label=f'F1 Score of {model_name}')

    ax.set_ylabel('Scores')
    ax.set_title(f"Comparison of Precision, Recall, F1 Score for {graph_name}")
    ax.set_xticks(np.arange(num_models))
    ax.set_xticklabels(model_names)
    ax.legend()

    plt.axis([-width, num_models - 1 + width, 0, 1])
    plt.grid(True)
    plt.savefig(fname=os.path.join(plot_path, f"prf_comparison_{graph_name}.png"))
    plt.show()