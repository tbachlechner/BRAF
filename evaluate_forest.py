import argparse
import pandas as pd
from trees import *
from utils import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--forest', default='BRAF', type=str, help='Type of random forest classifier to evaluate (BRAF/RF), default: BRAF')
parser.add_argument('--k', default = 10, type = int, help = 'k-nearest neighbors in BRAF parameter')
parser.add_argument('--folds', default = 10, type = int, help = 'Folds of data to use in k-fold cross validation')
parser.add_argument('--p_ratio', default = 0.5, type = float, help = 'p_ratio parameter of BRAF. 0 corresponds to RF')
parser.add_argument('--size', default = 100, type = int, help = 'Number of trees in forest')
args = parser.parse_args(); 

print(args)

path = 'diabetes.csv'

df = pd.read_csv(path)

full_data = df.values.tolist()

if args.forest == 'BRAF':
    classifier = BRAF(size = args.size, p_ratio = args.p_ratio, k = args.k)
    k_fold_evaluation(data = full_data, classifier = classifier, name = './data/BRAF',folds = args.folds)
elif args.forest == 'RF':
    classifier = RandomForest(size = args.size)
    k_fold_evaluation(data = full_data, classifier = classifier, name = './data/RF')

