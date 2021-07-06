import io
import requests
import numpy as np
import pandas as pd

from sklearn import datasets

def load_scikit_dataset(dataset):
    
    if dataset == 'iris':
        data = datasets.load_iris()
    elif dataset == 'wine':
        data = datasets.load_wine()
    else:
        print('É preciso selecionar um dataset disponivel')
        return None, None
    
    df = pd.DataFrame(data= np.c_[data['data'], data['target']],
                         columns= list(data['feature_names']) + ['target'])

    X = df[sorted(list((set(data['feature_names'])) - set(['target'])))]
    
    # convert to -1 and 1 to hinge loss
    #y = pd.get_dummies(df['target'])#.astype(int).replace(0, -1)
    y = df['target']
    
    return X, y
    
def load_iris():
    #source: https://archive.ics.uci.edu/ml/datasets/iris
    return load_scikit_dataset('iris')

def load_wine():
    #source: https://archive.ics.uci.edu/ml/datasets/wine
    return load_scikit_dataset('wine')

def load_breast_cancer():
    #source: archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
    url='https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
    s = requests.get(url).content

    columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', \
    'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'target']
    df = pd.read_csv(io.StringIO(s.decode('utf-8')), names=columns)

    X = df[sorted(list((set(df.columns)) - set(['target', 'Sample code number'])))]
    
    X.loc[:,'Bare Nuclei'] = X.loc[:,'Bare Nuclei'].replace('?', -1)

    # convert to -1 and 1 to hinge loss
    y = df['target']#pd.get_dummies(df['target'])#.astype(int).replace(0, -1)
    
    return X, y

def load_glass():
    #source: https://archive.ics.uci.edu/ml/datasets/glass+identification
    url='https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
    s = requests.get(url).content
    
    columns = ['Id', 'Clump_thickness', 'Uniformity_cell_size', 'Uniformity_cell_shape', 'Marginal_adhesion', \
                       'Single_e_cell_size', 'Bare_nuclei', 'Bland_chromatin', 'Normal_nucleoli', 'Mitoses', 'target']
    df = pd.read_csv(io.StringIO(s.decode('utf-8')), names=columns)
    
    X = df[sorted(list((set(df.columns)) - set(['target'])))]
    
    # convert to -1 and 1 to hinge loss
    y = df['target']#pd.get_dummies(df['target'])#.astype(int).replace(0, -1)
    
    return X, y

def load_skin():
    #source: https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation
    url='https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt'
    s = requests.get(url).content
    
    columns = ['B', 'G', 'R', 'target']
    df = pd.read_csv(io.StringIO(s.decode('utf-8')), names=columns, delimiter='\t')
    
    X = df[sorted(list((set(df.columns)) - set(['target'])))]
    
    # convert to -1 and 1 to hinge loss
    y = df['target']#pd.get_dummies(df['target'])
    
    return X, y

def load_statlog_shuttle():
    #source: https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)
    url='https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.tst'
    s = requests.get(url).content
    
    columns = ['time', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'target']
    df = pd.read_csv(io.StringIO(s.decode('utf-8')), names=columns, delimiter=' ')
    df.drop(['time'], axis=1, inplace=True)
    df = df.loc[df['target'].isin([1,4,5])]

    X = df[sorted(list((set(df.columns)) - set(['target'])))]
    
    # convert to -1 and 1 to hinge loss
    y = df['target']#pd.get_dummies(df['target'])
    
    return X, y