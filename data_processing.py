# Imports
import random

class k_fold_split:
    '''
    Splits full dataset into train/test sets via k folds (no replacement).
    split.train_test(k) calls the k-th fold

    Parameters:
    - data (List) in format data[0] = [features, Label]
    - folds (int) number of folds

    Usage:
    split = k_fold_split(data,folds)
    train_data, test_data = split.train_test(fold)
    '''

    def __init__(self, data, k):
        n_samples = len(data)
        self.k = k
        indices = list(range(0,n_samples))
        random.shuffle(indices)
        samples_fold = round(n_samples / k)
        self.split_data = []

        for i in range(k):
            fold = []
            j = 0
            while j < samples_fold and len(indices)>0:
                fold.append(data[indices.pop()])
                j +=1
            self.split_data.append(fold)


    def train_test(self,i):
        train = []
        test = []
        for j in range(0,self.k):
            if i == j:
                test.extend(self.split_data[j])
            else:
                train.extend(self.split_data[j])
        return train, test

class mean_impute:
    '''
    Imputes all elements in data of value val with the mean of all other class
    elements.

    Parameters:
    - data (List) in format data[0] = [features, Label]
    - columns (List of int) containing the column indices to Impute
    - val (float) value to replace with mean

    Usage:
    imp = mean_impute(train_data,list(range(1,8)))
    imputed_test_data = imp.call(test_data)

    '''

    def __init__(self,data,columns, val = 0):
        transposed_data = list(map(list, zip(*data)))
        self.means = []
        self.columns = columns
        self.val = val
        for i in columns:
            column = transposed_data[i]
            mean = 0
            no_val = [el for el in column if el != val]
            length_no_val = len(no_val)
            self.means.append(sum(no_val)/length_no_val)


    def call(self,data):
        transposed_data = list(map(list, zip(*data)))
        for i,j in enumerate(self.columns):
            column = transposed_data[j]
            mean = 0
            no_val = [el for el in column if el != self.val]
            for j, el in enumerate(column):
                if el == self.val:
                    column[j] = self.means[i]
        return list(map(list, zip(*transposed_data)))

class normalize:
    '''
    Normalizes all elements in columns to lie between 0 and 1.

    Parameters:
    - data (List) in format data[0] = [features, Label]
    - columns (List of int) containing the column indices to Impute

    Usage:
    norm = normalize(train_data,list(range(1,8)))
    normalized_test_data = norm.call(test_data)

    '''

    def __init__(self,data,columns):
        transposed_data = list(map(list, zip(*data)))
        self.mins = [min(column) for column in transposed_data]
        self.maxs = [max(column) for column in transposed_data]
        self.columns = columns

    def call(self,data):
        transposed_data = list(map(list, zip(*data)))
        for i, column in enumerate(self.columns):
            row = transposed_data[column]
            for j, el in enumerate(row):
                row[j] = (el-self.mins[column])/(self.maxs[column]-self.mins[column])
        return list(map(list, zip(*transposed_data)))
