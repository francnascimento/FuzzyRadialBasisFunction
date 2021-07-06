import numpy as np
import pandas as pd

from skfuzzy import cmeans, cmeans_predict
from sklearn.base import BaseEstimator, ClassifierMixin

class RFBNN(BaseEstimator, ClassifierMixin):
    def __init__(self, lr=10e-4, epochs=10000, centroids=5, m=2, random_state=42):
        self.lr = lr
        self.epochs = epochs
        self.centroids = centroids
        self.m = m
        self.random_state = random_state
        
        self.cmeans_model = None
        self.sigma = None
        self.wo = None
        self.bo = None
        self.error_cost = None
        
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"lr": self.lr, "epochs": self.epochs, "centroids": self.centroids, "m": self.m}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def train_cmeans(self, X, centroids, m=2, error=1e-5, maxiter=100, seed=42):
    
        # Calculating centroids
        cmeans_args = {
            'c': centroids,
            'm':m,
            'error': error,
            'maxiter': maxiter,
            'seed': seed
        }
        centroids, W, _, distances, _, _, pc = cmeans(X.T, **cmeans_args)
        cmeans_args['cntr_trained'] = centroids
        cmeans_args.pop('c', None)

        # Calculating sigma from training sample
        ms = np.unique(distances.T.argmin(axis=1), return_counts=True)[1]
        distances.T[distances.T > np.expand_dims(distances.T.min(axis=1), axis=1)] = 0
        sigma = np.sqrt((distances.T**2).sum(axis=0)/ms)

        return cmeans_args, sigma, W, pc

    def fuzzy_radial_basis_function(self, X, sigma, cmeans_args):
        _, _, distances, _, _, _ = cmeans_predict(X.T, **cmeans_args)
        z = np.exp( -(distances.T**2 / (2*(sigma**2))) )
        return z
    
    def fit(self, X, y):
        np.random.seed(self.random_state)
        
        feature_set = X
        one_hot_labels = pd.get_dummies(y).values
        centers = self.centroids
        lr = self.lr

        self.cmeans_model, self.sigma, u, _ = self.train_cmeans(feature_set, centroids=self.centroids, m=self.m, maxiter=1000, seed=self.random_state)
        
        attributes = feature_set.shape[1]
        output_labels = one_hot_labels.shape[1]

        self.wo = np.random.rand(centers, output_labels)
        self.bo = np.random.randn(output_labels) # bias

        self.error_cost = []
        for epoch in range(self.epochs):
        ############# feedforward

            # Phase 1
            z = self.fuzzy_radial_basis_function(feature_set, self.sigma, self.cmeans_model)
            zo = np.dot(z, self.wo) + self.bo

            ao = self.softmax(zo)

        ########## Back Propagation

        ########## Phase 1

            dcost_dzo = ao - one_hot_labels
            dzo_dwo = z

            dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

            dcost_bo = dcost_dzo

            # Update Weights ================

            self.wo -= self.lr * dcost_wo
            self.bo -= self.lr * dcost_bo.sum(axis=0)

            if epoch % 1000 == 0:
                loss = np.sum(-one_hot_labels * np.log(ao))
                self.error_cost.append(loss)

        return self
    
    def predict(self, X):
        
        feature_set = X
        z = self.fuzzy_radial_basis_function(feature_set, self.sigma, self.cmeans_model)
        zo = np.dot(z, self.wo) + self.bo
        ao = self.softmax(zo)
        
        y_pred = np.argmax(ao, axis=1)
        return y_pred
    
    def softmax(self, A):
        expA = np.exp(A)
        return expA / expA.sum(axis=1, keepdims=True)