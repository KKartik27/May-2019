# -*- coding: utf-8 -*-
#Refer to FetureEngg_PCA pdf for the same values

from numpy import array
import numpy as np
from sklearn import decomposition
from numpy.linalg import eig
from sklearn.preprocessing import StandardScaler
# define matrix
A = array([[10, 20, 10], [2, 5, 2], [8, 17, 7], [9, 20, 10], [12, 22, 11]])
print(A)
# calculate eigendecomposition
values, vectors = eig(A)
print(values)
print(vectors)

################
A = array([[10, 20, 10], [2, 5, 2], [8, 17, 7], [9, 20, 10], [12, 22, 11]])
print(A)
#Co Variance Matrix
#Co Variance(X1,X2,X3) = Corr(X1,X2,X3)*std(x1)*std(x2)*std(x3)
cov_Matrix = np.cov(A.T)
print(cov_Matrix)

#eigen values and vectors
values, vectors = eig(cov_Matrix)
print(values)
#Eigen Vector of Cov-Matrix is nothing but new PCA
print(vectors)

#Same above work is done simply by decomposition
pca = decomposition.PCA(n_components = 2)
pca.fit(A)
pca.explained_variance_
pca.explained_variance_ratio_
pca.explained_variance_ratio_.cumsum()  
X_std = StandardScaler().fit_transform(A)
print(X_std)