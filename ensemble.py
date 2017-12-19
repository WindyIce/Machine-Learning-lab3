import pickle
import numpy as np

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        
        # list of weak classifier and the weight for every weak classifier
        self.weak_clf = []
        self.a_clf = []
        pass

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Returns:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)
        
        # weights of every train samples, initilize with 1/n_samples
        w_samples = np.ones((n_samples,1)) / n_samples #(n, 1)
        
        for i in range(self.n_weakers_limit):
            # train a weak classifier.
            clf = self.weak_classifier(max_depth=6,min_samples_split=10,min_samples_leaf=10,random_state=1010)
            clf.fit(X, y, sample_weight=w_samples.reshape(-1))
            
            # calculate the error rate.
            y_predict = clf.predict(X).reshape(-1, 1)
            error_rate = np.sum(w_samples * (y_predict != y))
            
            # if weak_clf[i] is a bad classifier, get over it.
            if error_rate > 0.5:
                print('error_rate larger than 0.5')
                continue
            
            # else, calculate the importance score of ith classifier.
            importance_score = 0.5 * np.log((1-error_rate) / error_rate)
            
            # update the weights of every samples.
            weight_new = w_samples * np.exp(-error_rate * y * y_predict)
            normalized_iterm = np.sum(weight_new)
            w_samples = weight_new / normalized_iterm
            
            # store this weak classifier.
            self.weak_clf.append(clf)
            self.a_clf.append(importance_score)

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        n_samples = X.shape[0]
        n_classifier = len(self.weak_clf)
        score = np.zeros((n_samples, 1))
        for i in range(n_classifier):
            score_i = self.weak_clf[i].predict(X).reshape(n_samples, 1)
            score += self.a_clf[i] * score_i
        return score
        

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        score = self.predict_scores(X)
        y_pred = np.ones((X.shape[0], 1))
        y_pred[score < threshold] = -1
        
        return y_pred

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
