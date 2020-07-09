# Imports
import random
import math


# Implement BRAF
class BRAF:
    '''
    Biased Random Forest
    Parameters:
    - size (int) = 100: number of trees
    - p_ratio (float) = 0.5: Fraction of trees in biased forest
    - k (int) = 10: Number of nearest neighbors to minority class to use in
      critical tree
    - data in format data[0] = [features, Label]

    Usage:
    classifier = BRAF(size = 100,p_ratio = 0.5, k = 10)
    classifier.build_forest(data)
    predictions = classifier.predict(test_data)

    Ref:
    https://ieeexplore.ieee.org/document/8541100
    '''

    def __init__(self, size = 100, p_ratio = 0.5, k = 10):

        self.size = size
        self.p_ratio = p_ratio
        self.k = k


    def build_forest(self, data):
        '''
        Builds the forest to fit data
        '''

        self.classes = []
        sets = [], []
        for row in data:
            label = row[-1]
            if label not in self.classes:
                self.classes.append(row[-1])
            if label == self.classes[0]:
                sets[0].append(row)
            else:
                sets[1].append(row)

        majority_index = 0
        if len(sets[1]) > len(sets[0]):
            majority_index = 1

        T_maj = sets[majority_index]
        T_min = sets[1-majority_index]
        L_maj = T_maj[0][-1]
        L_min = T_min[0][-1]
        T_c = []
        for element in T_min:
            T_c.append(element)
            T_nn = self.k_nearest(T_maj, element, self.k, exclude_last_n = 1)
            for nn_element in T_nn:
                if nn_element not in T_c:
                    T_c.append(nn_element)

        self.rf1 = RandomForest(size = round((1-self.p_ratio)*self.size))
        self.rf1.build_forest(data)
        self.rf2 = RandomForest(size = round(self.p_ratio*self.size))
        self.rf2.build_forest(T_c)

    def single_predict(self, data, probabilities = False):
        '''
        Returns a prediction from a single row of features
        '''

        predictions = self.predict_trees(data)
        if probabilities:
            return sum(predictions)/len(predictions)
        else:
            sorted_classes = self.classes.copy()
            sorted_classes.sort( key = lambda x: predictions.count(x))
            prediction = sorted_classes[-1]
            return prediction

    def predict_trees(self, data):
        '''
        Returns the predictions for all trees in the forest
        '''
        predictions = self.rf1.predict_trees(data)
        predictions.extend(self.rf2.predict_trees(data))
        return predictions

    def predict(self, data, probabilities = False):
        '''
        Returns a prediction from a list or single row of features
        '''
        if isinstance(data[0],list):
            out = []
            for element in data:
                out.append(self.single_predict( element,probabilities = probabilities))
            return out
        else:
            return self.single_predict(data,probabilities = probabilities)


    def l2_norm2(self,row1, row2, exclude_last_n = 0):
        '''
        Calculate the Euclidean distance between two vectors (excliding the last n coordinates)
        '''
        distance = 0.0
        for i in range(len(row1) - exclude_last_n):
            distance += (row1[i] - row2[i])**2
        return distance

    def k_nearest(self,vectors, vector, k, exclude_last_n = 0):
        '''
        k_nearest neighbors
        '''
        distances = []
        for train_row in vectors:
            dist = self.l2_norm2(vector, train_row, exclude_last_n = exclude_last_n)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(k):
            neighbors.append(distances[i][0])
        return neighbors

# Implement Random Forest
class RandomForest:
    '''
    Random forest
    Parameters:
    - size (int): number of trees
    - fraction (int): fraction of training set used for training each tree
    - node_depth (int): maximum tree depth
    - n_search_pts (int): number of points used to search for optimal decision threshold
    - max_features_per_node (int): maximum features each node is allowed to use
    - data in format data[0] = [features, Label]

    Usage:

    classifier = RandomForest(size = 100)
    classifier.build_forest(data)
    predictions = classifier.predict(test_data)

    Ref:
    Inspired in part by
    https://machinelearningmastery.com/implement-random-forest-scratch-python/
    '''

    def __init__(self, size = 100, bagging_frac = .64, max_depth = 100, min_size = 1, classifier = 'gini', gamma = 1):
        self.trees = [my_tree(max_depth = max_depth, min_size = min_size,classifier = classifier, gamma = gamma) for _ in range(size)]
        self.bagging_frac = bagging_frac

    def select_n_random(self, List, n = None):
        '''
        Returns n random elements of List with replacement
        '''
        length = len(List)
        indices = [random.randrange(0,len(List)) for _ in range(0,n ) ]

        if n == None:
            return List_copy
        else:
            return [List[i] for i in indices]

    def build_forest(self, data):
        '''
        Builds the forest to fit data
        '''
        self.classes = []
        for row in data:
            if row[-1] not in self.classes:
                self.classes.append(row[-1])
        n_bag = round(self.bagging_frac * len(data))
        for tree in self.trees:
            tree.build_tree(self.select_n_random(data,n_bag))

    def single_predict(self, data, probabilities = False):
        '''
        Returns a prediction from a single row of features
        '''
        predictions = self.predict_trees(data)

        if probabilities:
            return sum(predictions)/len(predictions)
        else:
            sorted_classes = self.classes.copy()
            sorted_classes.sort( key = lambda x: predictions.count(x))
            prediction = sorted_classes[-1]
            return prediction

    def predict(self, data, probabilities = False):
        '''
        Returns a prediction from a list or single row of features
        '''
        if isinstance(data[0],list):
            out = []
            for element in data:
                out.append(self.single_predict( element,probabilities = probabilities))
            return out
        else:
            return self.single_predict(data,probabilities = probabilities)

    def predict_trees(self, data):
        '''
        Returns the predictions for all trees in the forest
        '''
        predictions = [tree.predict(data) for tree in self.trees]
        return predictions


# Implement Tree
class my_tree:
    def __init__(self,max_depth = 100, min_size = 1, classifier = 'gini', gamma = 1):
        self.max_depth = max_depth
        self.min_size = min_size
        self.classifier = classifier
        self.gamma = gamma #option for modified entropy

    def test_split(self,index, value, dataset):
        '''
        Split a dataset based on an attribute and an attribute value
        '''
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def gini_index(self,groups, classes):
        '''
        Calculate the Gini index for a split dataset
        '''
        # count all samples at split point
        n_instances = sum([len(group) for group in groups])
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = len(group)
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    def entropy_index(self, groups, classes, gamma = 1):
        '''
        Calculate the modified entropy index for a split dataset
        '''
        # count all samples at split point
        n_instances = sum([len(group) for group in groups])
        entropy = 0.0
        for group in groups:
            size = len(group)
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                if p>0:
                    score -= p * math.log(p,2)
            # weight the group score by its relative size
            entropy += score* (size / n_instances)
            entropy = entropy**(1/gamma)
        return entropy

    def cost(self,groups, classes):
        if self.classifier == 'gini':
            return self.gini_index(groups, classes)
        elif self.classifier == 'entropy':
            return self.entropy_index(groups, classes, gamma = self.gamma)

    def get_split(self,dataset):
        '''
        Select the best split point for a dataset
        '''
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                group_cost = self.cost(groups, class_values)
                if group_cost < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], group_cost, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    def to_terminal(self,group):
        '''
        Create a terminal node value
        '''
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def split(self,node,  depth):
        '''
        Create child splits for a node or make terminal
        '''
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        # check for max depth
        if depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal(left), to_terminal(right)
            return
        # process left child
        if len(left) <= self.min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], depth+1)
        # process right child
        if len(right) <= self.min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], depth+1)

    def build_tree(self,train):
        '''
        Build a decision tree
        '''
        self.root = self.get_split(train)
        self.split(self.root, 1)
        return self.root

    def print_tree(self,node, depth=0):
        '''
        Print a decision tree
        '''
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
            self.print_tree(node['left'], depth+1)
            self.print_tree(node['right'], depth+1)
        else:
            print('%s[%s]' % ((depth*' ', node)))

    def predict(self,row,node=None):
        '''
        Make a prediction with a decision tree
        '''
        if node == None:
            node = self.root
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict( row,node['left'])
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict( row,node['right'])
            else:
                return node['right']
