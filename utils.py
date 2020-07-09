import random
import matplotlib.pyplot as plt
from data_processing import *

def transpose_list(data):
    return list(map(list, zip(*data)))

def mean(data):
    return sum(data)/len(data)

def standard_error(data):
    length = len(data) 
    mean = sum(data) / length
    variance = sum([((x - mean) ** 2) for x in data]) / length
    out = variance ** 0.5 / (length ** 0.5)
    return out

def extract_targets(data):
    targets = []
    for el in data:
        targets.append(el[-1])
    return targets

        
def accuracy_precision_recall(predictions,targets):
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
        precision = None
    else:
        precision = true_positives / all_positives
    recall = true_positives / positives
    accuracy = correct/len(predictions)
    return accuracy, precision, recall

def fpr_tpr(predictions,targets):
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



def curve(prediction_probs, targets, kind = 'prc' , points = 100, print_plot = True,filename = None, k_fold = 0):
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
            
    [x,y] = transpose_list(output)
    
    

    if print_plot:
        assert filename != None, 'Provide a filename argument to curve().'
        plot = print_curve(x,y,filename = filename, kind = kind)
    
    return auc(x,y)

def eval_predictions(prediction_probs,threshold):
    predictions = []
    for i, prob in enumerate(prediction_probs):
        if prob>threshold:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions
    


# Compute area under curve using trapezoidal rule
def auc(x,y):
    area = 0

    for i in range(len(x)-1):
        area += abs((y[i]+y[i+1])/2* (x[i+1]-x[i]))
    return area

def evaluate(classifier, test_data, fold = 0, name = ''):

    targets = extract_targets(test_data)

    predictions = classifier.predict(test_data, probabilities = False)

    accuracy, precision, recall = accuracy_precision_recall(predictions,targets)

    prediction_probs = classifier.predict(test_data, probabilities = True)
    auroc = curve(prediction_probs, targets, kind = 'roc',print_plot = True, filename = name+'_roc_fold'+str(fold)+'.png')
    auprc = curve(prediction_probs, targets, kind = 'prc',print_plot = True, filename = name+'_prc_fold'+str(fold)+'.png')

    print('| Fold: {:2} | Accuracy: {:0.2f} | Precision: {:0.2f} | Recall: {:0.2f} | AUROC: {:0.2f} | AUPRC: {:0.2f} |'.format(fold,accuracy,precision,recall, auroc,auprc))
    return accuracy, precision, recall, auroc, auprc

def k_fold_evaluation(data, classifier,name, folds = 10):
    for i in range(len(data)):
        data[i][-1] = int(data[i][-1])


    split = k_fold_split(data,folds)

    accuracies = []
    precisions = []
    recalls = []
    aurocs = []
    auprcs = []
    print('-'*90)
    for fold in range(0,10):
        train_data, test_data = split.train_test(fold)


        imp = mean_impute(train_data,list(range(1,8)))
        norm = normalize(train_data,list(range(0,8)))

        ni_train_data =  norm.call(imp.call(train_data))
        ni_test_data =  norm.call(imp.call(test_data))


        classifier.build_forest(ni_train_data)

        accuracy, precision, recall, auroc, auprc = evaluate(classifier, ni_test_data, name = name,fold = fold)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        aurocs.append(auroc)
        auprcs.append(auprc)
    print('-'*90)
    print('')
    print('Averages +/- Standard Error')
    print('-'*27)
    print(' Accuracy:  {:0.2f} +/- {:0.2f}'.format(mean(accuracies), standard_error(accuracies)))
    print(' Precision: {:0.2f} +/- {:0.2f}'.format(mean(precisions), standard_error(precisions)))
    print(' Recall:    {:0.2f} +/- {:0.2f}'.format(mean(recalls), standard_error(recalls)))
    print(' AUROC:     {:0.2f} +/- {:0.2f}'.format(mean(aurocs), standard_error(aurocs)))
    print(' AUPRC:     {:0.2f} +/- {:0.2f}'.format(mean(auprcs), standard_error(auprcs)))

