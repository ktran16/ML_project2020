from sklearn.metrics import confusion_matrix
import itertools
import numpy as np

import matplotlib.pyplot as plt


# ploting functions
def loss_plot(model_name, train_loss, val_loss):
    plt.plot(train_loss, label='Training Loss')  
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss plot of {}'.format(model_name))
    #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')  
    #plt.rcParams["figure.figsize"] = (15,10)
    plt.legend()
    plt.show() 

def acc_plot(model_name, train_acc, val_acc):
    plt.plot(train_acc, label='Training Accuracy')  
    plt.plot(val_acc,label='Validation Accuracy')
    plt.title('Accuracy plot of {}'.format(model_name))
    #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')  
    #plt.rcParams["figure.figsize"] = (15,10)
    plt.legend()
    plt.show()


def plot(model_name, train_loss_h, train_acc_h, val_loss_h, val_acc_h):
    loss_plot(model_name, train_loss_h, val_loss_h)
    acc_plot(model_name, train_acc_h, val_acc_h)
        

        
def all_model_plot(all_models_val_loss, all_labels, title):
    for i in range(len(all_models_val_loss)):
        plt.plot(all_models_val_loss[i], label = all_labels[i])
    plt.title(title)
    #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')  
    plt.legend()
    plt.rcParams["figure.figsize"] = (15,10)
    plt.show()
    
    

# plot confusion matrix     
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=False):
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

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.rcParams["figure.figsize"] = (15,10)
    plt.show()