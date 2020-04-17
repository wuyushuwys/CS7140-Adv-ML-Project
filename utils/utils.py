from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.metrics import f1_score as _f1_score
from torch import optim
from time import time


def f1_score(y_true, y_pred):  # y-true: did not pass ReLu; y_pred: output
    return _f1_score(y_true.to('cpu'), y_pred.to('cpu'), average='weighted')


def fusion_test(model, inputs, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad(): 
        init = True
        for data in inputs:
            X_eeg = data[0].to(device)
            X_ecg = data[1].to(device)
            X_gsr = data[2].to(device)
            y = data[-1].to(device)
            outputs = model(X_eeg, X_ecg, X_gsr)
            total_loss += criterion(outputs, y)
            if init:
                y_true = y
                y_pred = outputs
                init = False
            else:
                y_true = torch.cat([y_true, y], dim=0)
                y_pred = torch.cat([y_pred, outputs], dim=0)
        _, y_pred = torch.max(y_pred, 1)
        outputs = (y_pred, y_true)
        accuracy  = (y_true == y_pred).sum().item()/len(y_pred)            
    return float(total_loss/len(inputs)), f1_score(y_true, y_pred), accuracy, outputs


def model_test(model, inputs, device, data_index, flag="test"):
    # model.eval()
    total_loss = 0
    with torch.no_grad(): 
        init = True
        for data in inputs:
            X, y = data[data_index].to(device), data[-1].to(device)
            outputs = model(X)
            total_loss += criterion(outputs, y)
            if init:
                y_true = y
                y_pred = outputs
                init = False
            else:
                y_true = torch.cat([y_true, y], dim=0)
                y_pred = torch.cat([y_pred, outputs], dim=0)

        outputs = y_pred
        _, y_pred = torch.max(y_pred, 1)
        accuracy  = (y_true == y_pred).sum().item()/len(y_pred)
        
        if flag=='test':
            return float(total_loss/len(inputs)), accuracy, f1_score(y_true, y_pred)
        elif flag=='eval':
            return accuracy, f1_score(y_true, y_pred), outputs.to('cpu'), y_true.to('cpu')  
        

def tokenize(data):
    criterion = 5
    if data[0]>criterion  and data[1]>criterion: # HVHA
        label = 0
        name = 'HAHV'
    elif data[0]>criterion  and data[1]<=criterion: # HVLA
        label = 1
        name = 'HALV'
    elif data[0]<=criterion  and data[1]>criterion: # LVHA
        label = 2
        name = 'LAHV'
    elif data[0]<=criterion  and data[1]<=criterion: # LVLA
        label = 3
        name = 'LALV'
    return label, name


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    plt.savefig('fusion_confusion_matrix.png')