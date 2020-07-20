# Usage: python evaluate_forest.py --forest BRAF --k 10 --folds 10 --p_ratio 0.5 --size 100 --print_plot --evaluate_on_train_set --k_fold_cross_val

# Imports
import argparse
import pandas as pd
from trees import *
from utils import *

# main()
def main(args):
    '''
    Imports data, and evaluates BRAF/RF classifiers via k-fold validation
    '''
    # Get data
    df = pd.read_csv(args.path)
    full_data = df.values.tolist()
    print('\n Length of full dataset:                       ', len(full_data))


    initial_split = k_fold_split(full_data,6)
    train_data, test_data = initial_split.train_test(0)
    print(' Initial train/test split into sets of length: ', len(train_data),'/',len(test_data))

    # Evaluate forests via k-fold validation

    print('\n Perform {:2}-fold cross validation within train set:'.format(args.folds))

    if args.forest == 'BRAF':
        print(' Model: Biased Random Forest. Parameters: size = {:3}, p = {:0.2}, k = {:2}'.format(args.size,args.p_ratio, args.k))
        classifier_val = BRAF(size = args.size, p_ratio = args.p_ratio, k = args.k)
        classifier_final = BRAF(size = args.size, p_ratio = args.p_ratio, k = args.k)
        name = './data/BRAF'
        if args.k_fold_cross_val:
            k_fold_evaluation(data = train_data, classifier = classifier_val, name = name,folds = args.folds, print_plot = args.print_plot)
    elif args.forest == 'RF':
        print(' Model: Plain Random Forest. Parameters: size = {:3}'.format(args.size))
        classifier_val = RandomForest(size = args.size)
        classifier_final = RandomForest(size = args.size, p_ratio = args.p_ratio, k = args.k)
        name = './data/RF'
        if args.k_fold_cross_val:
            k_fold_evaluation(data = train_data, classifier = classifier_val, name = name, folds = args.folds, print_plot = args.print_plot)

    print('\n Train model on full train set.')
    imp = mean_impute(train_data,list(range(1,8)))
    norm = normalize(train_data,list(range(0,8)))
    ni_train_data =  norm.call(imp.call(train_data))
    ni_test_data =  norm.call(imp.call(test_data))
    classifier_final.build_forest(ni_train_data)

    if args.evaluate_on_train_set:
        print('\n Model evaluation on train set:')
        accuracy, precision, recall, auroc, auprc, roc, prc = evaluate(classifier_final, ni_train_data, name = name,fold = '_final_train', print_plot = True)

    print('\n Model evaluation on test set:')
    accuracy, precision, recall, auroc, auprc, roc, prc = evaluate(classifier_final, ni_test_data, name = name,fold = '_final_test_', print_plot = True)



# Test for __main__
if __name__ == "__main__":
    # Implement argument parser
    parser = argparse.ArgumentParser(description='Evaluate Forest classifiers')
    parser.add_argument('--forest', default='BRAF', type=str, help='Type of random forest classifier to evaluate (BRAF/RF), default: BRAF')
    parser.add_argument('--path', default='diabetes.csv', type=str, help='path to data')
    parser.add_argument('--k', default = 10, type = int, help = 'k-nearest neighbors in BRAF parameter')
    parser.add_argument('--folds', default = 10, type = int, help = 'Folds of data to use in k-fold cross validation')
    parser.add_argument('--p_ratio', default = 0.5, type = float, help = 'p_ratio parameter of BRAF. 0 corresponds to RF')
    parser.add_argument('--size', default = 100, type = int, help = 'Number of trees in forest')
    parser.add_argument('--print_plot', action='store_true', help = 'Print PRC/ROC curves for each fold')
    parser.add_argument('--k_fold_cross_val', action='store_true', help = 'Perform k-fold cross validation within training set')
    parser.add_argument('--evaluate_on_train_set', action='store_true', help = 'Evaluate model on training set')

    args = parser.parse_args();

    main(args)
