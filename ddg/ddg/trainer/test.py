#%%
import numpy as np
import statsmodels.api as sm

# 가상의 데이터 생성
np.random.seed(1)
X = np.random.randn(100)
Y = np.random.randint(1,5, size=100)
Z = np.random.randint(1,5, size=100)

# Z를 통제한 X와 Y의 잔차 계산
X_residual = sm.OLS(X, sm.add_constant(Z)).fit().resid
Y_residual = sm.OLS(Y, sm.add_constant(Z)).fit().resid

# 잔차들의 상관계수 계산
conditional_correlation = np.corrcoef(X_residual, Y_residual)[0, 1]

print("조건부 상관계수: ", conditional_correlation)
#%%
icc = pg.intraclass_corr(data=X, targets=Y, raters=Y, ratings=X)

# %%
import pandas as pd
import numpy as np
import pingouin as pg

# Assuming 'data' is a pandas DataFrame with your data points as rows and features as columns
# And 'labels' is a pandas Series or a list with the class labels for each data point
data=X
labels=Y
# Create a DataFrame to hold the correlation coefficients
correlation_df = pd.DataFrame(columns=['Feature1', 'Feature2', 'Correlation', 'Class'])

# Iterate over classes and calculate correlations for each class
for class_label in np.unique(labels):
    # Select data points belonging to the current class
    class_data = data[labels == class_label]
    
    # Calculate correlation coefficients between all pairs of features for this class
    for i, feature1 in enumerate(class_data):
        for feature2 in class_data.columns[i+1:]:
            corr = class_data[feature1].corr(class_data[feature2])
            correlation_df = correlation_df.append({
                'Feature1': feature1,
                'Feature2': feature2,
                'Correlation': corr,
                'Class': class_label
            }, ignore_index=True)

# Now, for each pair of features, calculate the ICC across all classes
for (feature1, feature2), group in correlation_df.groupby(['Feature1', 'Feature2']):
    icc = pg.intraclass_corr(data=group, targets='Correlation', raters='Class', ratings='Feature1')
    print(f'ICC for features {feature1} and {feature2}: {icc["ICC"].values[0]}')
# %%
import scipy.stats
scipy.stats.pearsonr(X,Y)
# %%
class_1 = X[Y==1]
scipy.stats.pearsonr(class_1,Y[Y==1])
print(class_1)
print(Y[Y==1])
# %%
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
class_1 = 
nb_classifier.fit(X[], Y)


#%%

np.random.seed(1)
X = np.random.randn(100,7)
Y = np.random.randint(1,5, size=100)
Z = np.random.randint(1,5, size=100)

# %%
import minepy
mine = minepy.MINE()
mine.compute_score(X, Y)
def print_stats(mine):
    print ("MIC", mine.mic())
    print( "MAS", mine.mas())
    # print "MEV", mine.mev()
    # print "MCN (eps=0)", mine.mcn(0)
    # print "MCN (eps=1-MIC)", mine.mcn_general()
    # print "GMIC", mine.gmic()
    # print "TIC", mine.tic()
print(mine)
# %%
print(X.shape)
print(Y.shape)
# %%
Y = Y.reshape(-1,1)
# %%
from sklearn.cross_decomposition import CCA
cca = CCA(1)
cca.fit(X, Y)
cca.score(X, Y)
# %%
cca2 = CCA(2)
cca2.fit(X, Y)
cca2.score(X, Y)

# %%
np.random.seed(2)
Y2 = np.random.randint(1,5, size=100)
cca2 = CCA(1)
cca2.fit(X, Y2)
cca2.score(X, Y2)
# %%
