from sklearn import decomposition, preprocessing
import pandas as pd

#PCA without standardization
sqft = [1000, 600, 1200, 1400] #sft units
n_rooms = [3, 2, 2, 4] #Integer number
price = [1000000, 700000, 1200000, 1500000] #Currency $$
house_data = pd.DataFrame({'sqft':sqft, 'n_rooms':n_rooms, 'price':price})
print(house_data)
pca = decomposition.PCA()
pca.fit(house_data)
pca.explained_variance_
pca.explained_variance_ratio_
pca.explained_variance_ratio_.cumsum()

#PCA with standardization
scaler = preprocessing.StandardScaler()
scaler.fit(house_data)
scaled_house_data = scaler.transform(house_data)
print(scaled_house_data)
pca_s = decomposition.PCA()
pca_s.fit(scaled_house_data)
pca_s.explained_variance_
pca_s.explained_variance_ratio_
pca_s.explained_variance_ratio_.cumsum()