from __future__ import print_function
import time
import numpy as np
import pandas as pd
# from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# %matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

dataset = pd.read_csv('cm1.csv')
dataset.dropna(inplace=True)
# Print head
print(dataset)

print('Size of the dataframe: {}'.format(dataset.shape))


defect_true_false = dataset.groupby('defects')['b'].apply(lambda x: x.count())
print('False: ',defect_true_false[0])
print('True: ',defect_true_false[1])


feat_cols=[dataset.columns.values.tolist()][0]
feat_cols.remove("defects")
print("feat_cols ", feat_cols)


from sklearn.manifold import TSNE
# perplexity parameter can be changed based on the input datatset
# dataset with larger number of variables requires larger perplexity
# set this value between 5 and 50 (sklearn documentation)
# verbose=1 displays run time messages
# set n_iter sufficiently high to resolve the well stabilized cluster
# get embeddings
tsne_em = TSNE(n_components=2, perplexity=12.0, n_iter=3000, verbose=1).fit_transform(dataset)

from bioinfokit.visuz import cluster
cluster.tsneplot(score=tsne_em)

color_class = dataset['defects'].to_numpy()
cluster.tsneplot(score=tsne_em, colorlist=color_class, colordot=('#713e5a', '#63a375', '#edc79b', '#d57a66', '#ca6680', '#395B50', '#92AFD7', '#b0413e', '#4381c1', '#736ced'), 
    legendpos='upper right', legendanchor=(1.15, 1) )

from scipy.stats import chisquare
import scipy.stats as scs
def categories(series):
    return range(int(series.min()), int(series.max()) + 1)
def chi_square_of_df_cols(df, col1, col2):
    df_col1, df_col2 = df[col1], df[col2]

    result = [[sum((df_col1 == cat1) & (df_col2 == cat2))
               for cat2 in categories(df_col2)]
              for cat1 in categories(df_col1)]

    return scs.chi2_contingency(result)

# print(chi_square_of_df_cols(dataset, 'loc', 'defects'))
# from scipy.stats import chi2_contingency
# chi2,p,dof,expec = chi2_contingency(dataset['v(g)'],dataset['defects'])
# print("chi2 ",chi2,"else ",p,dof,expec)

import sklearn.feature_selection
print(sklearn.feature_selection.chi2(dataset,dataset['defects'])[0])

chi_square_vals= sklearn.feature_selection.chi2(dataset,dataset['defects'])
# chi_square_vals=sorted(chi_square_vals)

print(chi_square_vals)


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# test=sklearn.feature_selection.SelectKBest(score_func=sklearn.feature_selection.chi2, k=10)
# X_new=test.fit_transform(dataset[feat_cols], dataset['defects'])
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X_new, dataset['defects'], test_size=0.9, random_state=1)

clf = RandomForestClassifier(max_depth=5, random_state=0)
out=clf.fit(dataset[feat_cols], dataset['defects'] )

print(clf.predict([[1.1,1.4,1.4,1.4,1.3,1.3,1.3,1.3,1.3,1.3,1.3,1.3,2,2,2,2,1.2,1.2,1.2,1.2,1.4]]))
print(clf.predict([[11,3,1,1,49,215.22,0.07,14.67,14.67,3156.61,0.07,175.37,0,0,1,0,12,9,27,22,5]]))
print(clf.predict([[411,73,30,41,1500,12749.77,0.02,43.41,293.68,553518.63,4.25,30751.03,45,339,42,0,48,314,932,568,141]]))


# print("training set score ", out.score(x_train, y_train))

# print("test set score ", out.score(x_test, y_test))
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 
chi2_features = SelectKBest(chi2, k = 16) 
X_kbest_features = chi2_features.fit_transform(dataset[feat_cols], dataset['defects']) 
  
# Reduced features 
print('Original feature number:', dataset[feat_cols].shape[1]) 
print('Reduced feature number:', X_kbest_features.shape[1])

clf = RandomForestClassifier(max_depth=5, random_state=0)
out=clf.fit(X_kbest_features, dataset['defects'] )

# print(clf.predict([[1.1,1.4,1.4,1.4,1.3,1.3,1.3,1.3,1.3,1.3,1.3,1.3,2,2,2,2,1.2,1.2,1.2,1.2,1.4]]))
# print(clf.predict([[11,3,1,1,49,215.22,0.07,14.67,14.67,3156.61,0.07,175.37,0,0,1,0,12,9,27,22,5]]))
# print(clf.predict([[411,73,30,41,1500,12749.77,0.02,43.41,293.68,553518.63,4.25,30751.03,45,339,42,0,48,314,932,568,141]]))
