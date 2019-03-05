
# coding: utf-8

# # PYTHON Common Tools for Jupyter Notebooks

# In[4]:

import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib
import sklearn as sk
import numpy as np
from sklearn import preprocessing


# ## Plot Correlation Matrix
# 
# Parameters:
# *df - Pandas dataframe to pass
# *filename - should end with .png extension
# *size - 30 is a good number for a large matris

# In[5]:

#  This function does the actual graphical plotting of the correlation matrix.  To see the actual correlation 
#  numbers, simply use this call: df_scaled.corr()

def plot_corr(df, filename, size ):
    corr = df.corr()
    # 30 is a good number for size
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=70)
    plt.yticks(range(len(corr.columns)), corr.columns)
    fig.colorbar(cax)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
        ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    fig.savefig(filename)
    #plt.show()
    
    # Filename should end in .png
  


# ## Confusion Matrix
# 
# The below creates a graphical confusion matrix when the expected and predicted y values are passed with an array of target names.
# 
# Here's how I create the target array:
# 
# target_array = []
# test = df_t[41].drop_duplicates()  # where 41 is the column where the target vector resides
# for t in test:
#     target_array.append(t)
#     
# 

# In[4]:

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.colorbar()

    if normalize:
        
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, decimals=2)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.clim(0,1) # Reset the colorbar to reflect probabilities

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.grid('off')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def call_confusion_matrix(y_test, y_pred, target_array, filename, size=15):
    from sklearn.metrics import confusion_matrix
    #class_names = ['On-Time', 'Late']# Compute confusion matrix
    class_names = target_array
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    filename1 = ("1_%s" % filename)
    filename2 = ("2_%s" % filename)

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(size, size), dpi=200)
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
    
    plt.savefig(filename1, bbox_inches='tight')

    # Plot normalized confusion matrix
    plt.figure(figsize=(size, size), dpi=200)

    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
    
    plt.savefig(filename2, bbox_inches='tight')

    plt.show()


# ## Cross Validation Calls for Random Forest Classifier
# 
# I use this fairly often.  I call it like such:
# 
# cross_val_RF(model_rf, X_train, y_train)

# In[6]:

import operator 

def cross_val_RF(model, X, y):
    
    rs = RandomizedSearchCV(model, param_distributions={
        'n_estimators': stats.randint(30, 200),
        'max_features': ['auto', 'sqrt', 'log2'],
        "max_depth": [3, None],
        "max_features": stats.randint(1, 11),
        "min_samples_split": stats.randint(1, 11),
        "min_samples_leaf": stats.randint(1, 11),
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"]})

    rs.fit(X_train, y_train)
    
    report(rs.grid_scores_)  

    
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=operator.itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


# ## Data Enumeration
# 
# This is useful for automatically converting symbols to numbers for the purpose of classification.
# 
# '''
# Call like such:
# 
# columns_to_convert =[1,2,3,41]
# 
# for col in columns_to_convert:
#     enumerate_text(df, col)
#     
# '''

# In[7]:

def enumerate_text(df, col):
    target_array = []
    test = df[col].drop_duplicates()
    for t in test:
        target_array.append(t)

    p = [(j, i) for i, j in enumerate(target_array)]

    b = dict(p)
    print (b)

    df[col].replace(b, inplace=True)
    #df_t[:15]
    return df[col]



# ## Receiver Operating Characteristic (ROC) metric 
# 
# Used to evaluate classifier output quality using cross-validation.
# 
# ROC curves typically feature true positive rate on the Y axis, and false positive rate on the X axis. This means that the top left corner of the plot is the “ideal” point - a false positive rate of zero, and a true positive rate of one. This is not very realistic, but it does mean that a larger area under the curve (AUC) is usually better.
# 
# The “steepness” of ROC curves is also important, since it is ideal to maximize the true positive rate while minimizing the false positive rate.
# 
# This roughly shows how the classifier output is affected by changes in the training data, and how different the splits generated by K-fold cross-validation are from one another.

# In[2]:

def roc(model, X, y, target, filename):
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    
    from itertools import cycle
    from scipy import interp

    cv = StratifiedKFold(n_splits=6)
    
    y = np.array(y)
    X = np.array(X)
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2
    plt.figure(figsize=(10, 10), dpi=200)
    i = 0
    for (train, test), color in zip(cv.split(X, y), colors):
        probas_ = model.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,
             label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
         label='Luck')

    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(filename, bbox_inches='tight')
    plt.show()


# ## Principal Components Analysis
# 
# A method to take a large number of X features and re-define them along a smaller number of principal component axes.  I would use this to reduce dimensionality of a predictive set of X features down to a smaller, equally useful set of principal component vectors.

# In[3]:

from sklearn import decomposition
pca = decomposition.PCA()

def fit_features_pca(X):
    from sklearn import decomposition
    pca = decomposition.PCA()
    pca.fit(X)
    print("PCA variance by principal component\n", pca.explained_variance_)  
def reduce_features_pca(n, X):
    print("Shape of the original X matrix\n", X.shape)
    pca.n_components = n
    X_reduced = pca.fit_transform(X)
    print("Shape of the reduced X matrix\n", X_reduced.shape)
    return X_reduced

    


# In[ ]:



