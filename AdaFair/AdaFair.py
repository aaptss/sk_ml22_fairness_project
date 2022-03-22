import numpy as np
from tqdm.notebook import tqdm
from sklearn.tree import DecisionTreeClassifier

class AdaFair():
    """
    Generic class for construction of boosting models
    
    :param n_estimators: int, number of estimators (number of boosting rounds)
    :param base_classifier: callable, a function that creates a weak estimator. Weak estimator should support sample_weight argument
    :param get_alpha: callable, a function, that calculates new alpha given current distribution, prediction of the t-th base estimator,
                      boosting prediction at step (t-1) and actual labels
    :param get_distribution: callable, a function, that calculates samples weights given current distribution, prediction, alphas and actual labels
    """
    def __init__(self, sa_index, n_estimators=200, base_classifier=DecisionTreeClassifier(max_depth=1),
                 sa_label=1, epsilon=0):
        self.n_estimators = n_estimators
        self.base_classifier = base_classifier
        self.sa_index = sa_index
        self.sa_label = sa_label
        self.epsilon = epsilon
        
    def fit(self, X, y):
        n_samples = len(X)
        distribution = np.ones(n_samples, dtype=float) / n_samples
        
        self.classifiers = []
        self.alphas = []
        for i in tqdm(range(self.n_estimators)):
            # create a new classifier
            self.classifiers.append(self.base_classifier)     
            self.classifiers[-1].fit(X, y, sample_weight=distribution)
            
            # make a prediction
            h = np.sign(self.classifiers[-1].predict(X))
            h_hat = np.max(self.classifiers[-1].predict_proba(X), axis=1)
            
            # update alphas, append new alpha to self.alphas
            error_rate = np.sum(distribution * (y != h)) / np.sum(distribution)
            alpha = np.log((1 - error_rate) / error_rate) / 2
            self.alphas.append(alpha)
            
            sa_pos = (X[:, self.sa_index] == self.sa_label) * (y == 1)
            sa_neg = (X[:, self.sa_index] == self.sa_label) * (y == -1)
            nonsa_pos = (X[:, self.sa_index] != self.sa_label) * (y == 1)
            nonsa_neg = (X[:, self.sa_index] != self.sa_label) * (y == -1)            
            
            ahs = np.sign(np.sum(np.array(self.alphas) * np.array([clf.predict(X) for clf in self.classifiers]).T, axis=1))
            
#             dTPR = TPR(y, ahs, X, self.sa_index, self.sa_label, agg='non-prot') - TPR(y, ahs, X, self.sa_index, self.sa_label, agg='prot')
#             dTNR = TNR(y, ahs, X, self.sa_index, self.sa_label, agg='non-prot') - TPR(y, ahs, X, self.sa_index, self.sa_label, agg='prot')
            
#             u = (h != y) * (abs(dTPR) > self.epsilon) * nonsa_pos * (dTPR > 0) * abs(dTPR) \
#                 + (h != y) * (abs(dTPR) > self.epsilon) * sa_pos * (dTPR < 0) * abs(dTPR) \
#                 + (h != y) * (abs(dTNR) > self.epsilon) * nonsa_neg * (dTNR > 0) * abs(dTNR) \
#                 + (h != y) * (abs(dTNR) > self.epsilon) * sa_neg * (dTNR < 0) * abs(dTNR)
            
            dFNR = np.sum(ahs[sa_pos] != y[sa_pos]) / np.sum(sa_pos) - np.sum(ahs[nonsa_pos] != y[nonsa_pos]) / np.sum(nonsa_pos)
            dFPR = np.sum(ahs[sa_neg] != y[sa_neg]) / np.sum(sa_neg) - np.sum(ahs[nonsa_neg] != y[nonsa_neg]) / np.sum(nonsa_neg)
            
            u = (h != y) * (abs(dFNR) > self.epsilon) * nonsa_pos * (dFNR > 0) * abs(dFNR) \
                + (h != y) * (abs(dFNR) > self.epsilon) * sa_pos * (dFNR < 0) * abs(dFNR) \
                + (h != y) * (abs(dFPR) > self.epsilon) * nonsa_neg * (dFPR > 0) * abs(dFPR) \
                + (h != y) * (abs(dFPR) > self.epsilon) * sa_neg * (dFPR < 0) * abs(dFPR)
            
            # update distribution and normalize
            distribution = distribution * np.exp(alpha * h_hat * (y != h)) * (1 + u)
            C = np.sum(distribution)
            distribution = distribution / C

    
    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])        
        
        #get the weighted votes of the classifiers
        f = self.alphas[0] * self.classifiers[0].predict(X)
        for i in range(1, len(self.classifiers)):
            f += self.alphas[i] * self.classifiers[i].predict(X)
        out = np.sign(f)

        return out