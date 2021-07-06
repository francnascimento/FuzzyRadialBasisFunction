import numpy as np
import pandas as pd

from src.rfbnn import RFBNN
from src.metrics import classification_metrics

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

def hyperparameter_search(model, X, y, param_grid, cv=3):
    
    clf = GridSearchCV(model, param_grid = param_grid, cv = cv, scoring='accuracy', verbose=True, n_jobs=-1)
    
    best_clf = clf.fit(X, y)
    print("Best: %f using %s" % (best_clf.best_score_, best_clf.best_params_))
    
    return best_clf.best_params_

def train_logistic_regression(X, y, param_grid=None, cv=3, seed=42):
    
    Xtrain, Xtest, ytrain, ytest = train_test(X, y, split=0.7, seed=seed)

    if param_grid:
        model = LogisticRegression(random_state=seed)
        best_params = hyperparameter_search(model, Xtrain, ytrain, param_grid, cv)
        
        model = LogisticRegression(**best_params, random_state=seed).fit(Xtrain, ytrain)
    else:
        model = LogisticRegression(random_state=seed).fit(Xtrain, ytrain)

    ytest_pred = model.predict(Xtest)
    

    ytest_dummies = pd.get_dummies(ytest).to_numpy()
    ytest_pred_dummies = pd.get_dummies(ytest_pred).to_numpy()
    
    return classification_metrics(ytest_dummies, ytest_pred_dummies)

def train_multilayer_perceptron(X, y, param_grid=None, cv=3, seed=42):
    
    Xtrain, Xtest, ytrain, ytest = train_test(X, y, split=0.7, seed=seed)

    if param_grid:
        model = MLPClassifier(random_state=seed)
        best_params = hyperparameter_search(model, Xtrain, ytrain, param_grid, cv)
        
        model = MLPClassifier(**best_params, random_state=seed).fit(Xtrain, ytrain)
    else:
        model = MLPClassifier(random_state=seed).fit(Xtrain, ytrain)
        
    ytest_pred = model.predict(Xtest)
    
    ytest_dummies = pd.get_dummies(ytest).to_numpy()
    ytest_pred_dummies = pd.get_dummies(ytest_pred).to_numpy()

    return classification_metrics(ytest_dummies, ytest_pred_dummies)

def train_fuzzy_rbf_nn(X, y, param_grid=None, hyper=True, cv=3, seed=42):
    
    Xtrain, Xtest, ytrain, ytest = train_test(X, y, split=0.7, seed=seed)

    if hyper:
        model = RFBNN(random_state=seed)
        best_params = hyperparameter_search(model, Xtrain, ytrain, param_grid, cv)
        
        model = RFBNN(**best_params).fit(Xtrain, ytrain)
    else:
        model = RFBNN(**param_grid, random_state=seed).fit(Xtrain, ytrain)
    
    ytest_pred = model.predict(Xtest)
    
    ytest_dummies = pd.get_dummies(ytest).to_numpy()
    ytest_pred_dummies = pd.get_dummies(ytest_pred).to_numpy()

    return classification_metrics(ytest_dummies, ytest_pred_dummies)

def train_test(X, y, split=0.7, seed=42):

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=split, random_state=seed)
    
    #standardize the dataset
    scaler = MinMaxScaler()
    scaler.fit(Xtrain)

    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    return Xtrain, Xtest, ytrain, ytest