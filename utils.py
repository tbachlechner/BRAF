# Imports
import random
import matplotlib.pyplot as plt
from data_processing import *


# Define various helper functions
def transpose_list(data):
    '''
    Transpose first two levels of dataset

    Parameters:
    data (list(list))
    '''

    return list(map(list, zip(*data)))


def mean(data):
    '''
    Compute mean of data

    Parameters:
    data
    '''
    
    return sum(data)/len(data)


def standard_error(data):
    '''
    Compute standard error of data

    Parameters:
    data
    '''

    length = len(data)
    mean = sum(data) / length
    variance = sum([((x - mean) ** 2) for x in data]) / length
    out = variance ** 0.5 / (length ** 0.5)
    return out


def extract_targets(data):
    '''
    Extract target from data

    Parameters:
    data[0] = [Features, Label]
    '''

    targets = []
    for el in data:
        targets.append(el[-1])
    return targets


# Define various helper functions for evaluation
def accuracy_precision_recall(predictions,targets):
    '''
    Compute accuracy, precision and recall

    Parameters:
    predictions,targets
    '''

    assert len(predictions) == len(targets)
    true_positives = 0
    all_positives = 0
    positives = 0
    correct = 0

    for i in range(len(predictions)):
        prediction = predictions[i]
        target = targets[i]
        if prediction == 1:
            all_positives += 1
            if targets[i] == 1:
                true_positives+=1
        if target ==1:
            positives += 1
        if target == prediction:
            correct +=1
    if all_positives == 0:
        precision = 1
    else:
        precision = true_positives / all_positives
    recall = true_positives / positives
    accuracy = correct/len(predictions)
    return accuracy, precision, recall


def fpr_tpr(predictions,targets):
    '''
    Compute false positive rate and true positive rate

    Parameters:
    predictions, targets
    '''

    assert len(predictions) == len(targets)
    true_positives = 0
    false_positives = 0
    positives = sum(targets)
    negatives = len(targets)-positives
    for i in range(len(predictions)):
        prediction = predictions[i]
        target = targets[i]
        if prediction == 1:
            if targets[i] == 1:
                true_positives+=1
            elif targets[i] != 1:
                false_positives += 1

    tpr = true_positives / positives
    fpr = false_positives / negatives
    return fpr, tpr


def print_curve(x,y,filename,kind = 'roc'):
    '''
    Create nice plots from curve dataset and save them at filename
    '''

    if kind == 'prc':
        x_label = 'Recall'
        y_label = 'Precision'
        title = 'Precision Recall Curve. AUC: {:0.2}'.format(auc(x,y))
    elif kind == 'roc':
        x_label = 'False Positive Rate'
        y_label = 'True Positive Rate'
        title = 'ROC Curve. AUC: {:0.2}'.format(auc(x,y))
    fig, ax = plt.subplots()
    if kind == 'roc':
        plt.plot(range(0,2),range(0,2), '--',color='gray',  linewidth=1,alpha = .3)
    ax.plot(x,y,color = '#B31B1B')
    plt.ylim(0,1.01)
    plt.xlim(0,1)
    ax.set(xlabel=x_label, ylabel=y_label,
           title=title)
    ax.grid()
    fig.savefig(filename,  dpi=400)
    plt.close('all')


def curve(prediction_probs, targets, kind = 'prc' , points = 100, print_plot = True,
          filename = None, k_fold = 0):
    '''
    Return area under curve and also print curve for precision recall Curve
    and receiver operating characteristic curve.
    '''

    output = []
    for i in range(points+1):
        threshold = i/points
        predictions = eval_predictions(prediction_probs,threshold)
        if kind == 'prc':
            acc, precision, recall = accuracy_precision_recall(predictions,targets)
            if precision!= None:
                output.append( ( recall,precision ) )
        elif kind == 'roc':
            output.append(fpr_tpr(predictions,targets))
    if kind == 'prc':
        output.insert(0,(1,0))
    if kind == 'roc':
        output.insert(0,(1,1))

    [x,y] = transpose_list(output)

    if print_plot:
        assert filename != None, 'Provide a filename argument to curve().'
        plot = print_curve(x,y,filename = filename, kind = kind)

    return auc(x,y), [x,y]


def eval_predictions(prediction_probs,threshold):
    '''
    Compute binary predictions from prediction probabilies
    '''

    predictions = []
    for i, prob in enumerate(prediction_probs):
        if prob>threshold:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


def auc(x,y):
    '''
    Compute area under curve using trapezoidal rule
    '''

    area = 0

    for i in range(len(x)-1):
        area += abs((y[i]+y[i+1])/2* (x[i+1]-x[i]))
    return area


def evaluate(classifier, test_data, fold = 0, name = '', print_plot = True):
    '''
    Evaluate a classifier to return Accuracy, Precision, Recall, AUROC and AUPRC
    and print ROC and PRC.

    Parameters:
    classifier (classifier) : Classifier with method predict and argument probabilities
    test_data

    Usage:
    evaluate(classifier, test_data, name = name)
    '''

    targets = extract_targets(test_data)
    predictions = classifier.predict(test_data, probabilities = False)
    accuracy, precision, recall = accuracy_precision_recall(predictions,targets)
    prediction_probs = classifier.predict(test_data, probabilities = True)
    auroc, roc = curve(prediction_probs, targets, kind = 'roc',print_plot = print_plot, filename = name+'_roc_fold'+str(fold)+'.png')
    auprc, prc = curve(prediction_probs, targets, kind = 'prc',print_plot = print_plot, filename = name+'_prc_fold'+str(fold)+'.png')
    print('| Fold: {:2} | Accuracy: {:0.2f} | Precision: {:0.2f} | Recall: {:0.2f} | AUROC: {:0.2f} | AUPRC: {:0.2f} |'.format(fold,accuracy,precision,recall, auroc,auprc))
    return accuracy, precision, recall, auroc, auprc, roc, prc


def convert_to_averages(list_of_lists):
    '''
    Converts a list of shape [ [x1,y1], [x2,y2], ... ] to [ x_avg, y_avg ]
    '''
    transposed_x = transpose_list(transpose_list(list_of_lists)[0])
    transposed_y = transpose_list(transpose_list(list_of_lists)[1])
    out_x = [mean(x) for x in transposed_x]
    out_y = [mean(x) for x in transposed_y]
    return [out_x,out_y]


def k_fold_evaluation(data, classifier,name, folds = 10, print_plot = True):
    '''
    Train classifier on k-fold cross validation and evaluate to return Accuracy,
    Precision, Recall, AUROC and AUPRC and print ROC and PRC.

    Parameters:
    classifier (classifier) : Classifier with method predict and argument probabilities
    test_data

    Usage:
    evaluate(classifier, test_data, name = name)
    '''

    for i in range(len(data)):
        data[i][-1] = int(data[i][-1])
    split = k_fold_split(data,folds)
    accuracies = []
    precisions = []
    recalls = []
    aurocs = []
    auprcs = []
    rocs = []
    prcs = []
    print('-'*90)
    for fold in range(0,10):
        train_data, test_data = split.train_test(fold)
        imp = mean_impute(train_data,list(range(1,8)))
        norm = normalize(train_data,list(range(0,8)))
        ni_train_data =  norm.call(imp.call(train_data))
        ni_test_data =  norm.call(imp.call(test_data))
        classifier.build_forest(ni_train_data)
        accuracy, precision, recall, auroc, auprc, roc, prc = evaluate(classifier, ni_test_data, name = name,fold = fold, print_plot = print_plot)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        aurocs.append(auroc)
        auprcs.append(auprc)
        rocs.append(roc)
        prcs.append(prc)

    avg_roc = convert_to_averages(rocs)
    avg_prc = convert_to_averages(prcs)
    print_curve( x = avg_roc[0],y = avg_roc[1],filename = name+'_roc_avg.png', kind = 'roc')
    print_curve( x = avg_prc[0],y = avg_prc[1],filename = name+'_prc_avg.png', kind = 'prc')

    print('-'*90)
    print('')
    print('Averages +/- Standard Error')
    print('-'*27)
    print(' Accuracy:  {:0.2f} +/- {:0.2f}'.format(mean(accuracies), standard_error(accuracies)))
    print(' Precision: {:0.2f} +/- {:0.2f}'.format(mean(precisions), standard_error(precisions)))
    print(' Recall:    {:0.2f} +/- {:0.2f}'.format(mean(recalls), standard_error(recalls)))
    print(' AUROC:     {:0.2f} +/- {:0.2f}'.format(mean(aurocs), standard_error(aurocs)))
    print(' AUPRC:     {:0.2f} +/- {:0.2f}'.format(mean(auprcs), standard_error(auprcs)))
