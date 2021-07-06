import numpy as np

def classification_metrics(y, y_pred):
    n_samples, n_targets = y.shape
    
    n_0 = (y_pred.argmax(axis=1) == y.argmax(axis=1)).sum()/n_samples
    
    ms = np.unique(y.argmax(axis=1), return_counts=True)[1]
    aux, q_i = np.unique(y_pred.argmax(axis=1)[y_pred.argmax(axis=1) == y.argmax(axis=1)], return_counts=True)
        
    indexes_not_seted = list(set(list(range(n_targets))) - set(aux))

    for index in indexes_not_seted:
        q_i = np.insert(q_i, index, 0)

    n_a = sum(q_i/ms)/n_targets
    
    return n_a, n_0